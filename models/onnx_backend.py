"""ONNX-Runtime encoder backends for the eval pipeline.

Ported / distilled from the `serving-bench` branch
(`serving/backends/onnx_cuda.py` + `onnx_trt.py`). The serving harness wraps
everything in a `predict_batch` API for LitServe; here we only need the part
that swaps a GLiNER2 model's encoder for an ONNX-Runtime session, so that the
existing `evaluation/evaluate.py` `run_inference()` / `evaluate_model()`
machinery measures F1 + latency on the quantized variant unchanged.

The encoder is replaced with `_ORTEncoderWrapper`, a `torch.nn.Module` whose
`forward(input_ids=, attention_mask=)` returns a `BaseModelOutput` with
`.last_hidden_state` — exactly what `gliner2/model.py::_encode_batch` and
`gliner2/inference/engine.py` consume. The GLiNER2 task heads keep running in
PyTorch, so `model.extract(text, schema)` behaves identically.

Backends:
    pytorch_fp32   baseline — model unchanged (FP32)
    pytorch_fp16   model.half()
    onnx_cuda_fp16 encoder via ORT CUDA EP   (heads PyTorch FP16)
    onnx_trt_fp16  encoder via ORT TensorRT EP (heads PyTorch FP16)

Diagnostic env vars (shared names with serving-bench):
    SERVING_IOBINDING=1        enable IOBinding GPU path (default off)
    SERVING_LOG_TIMING=1       print encoder forward latency per call
    SERVING_TRT_OPT_LEVEL      TRT builder optimization level (default 1)
    SERVING_TRT_WORKSPACE_GB   TRT max workspace GB (default 2)
    SERVING_TRT_EXCLUDE_OPS    ops to keep off TRT (default Range,RandomNormalLike,ScatterND)
"""
import os
import sys
import time
from typing import List

import numpy as np
import torch

# Make pip-installed CUDA 12 / TensorRT libs visible to ORT (no-op off-Linux).
import models._cuda_libs  # noqa: F401

_LOG_TIMING = os.environ.get("SERVING_LOG_TIMING", "0") == "1"

ONNX_BACKENDS = ("onnx_cuda_fp16", "onnx_trt_fp16")
ALL_BACKENDS = ("pytorch_fp32", "pytorch_fp16") + ONNX_BACKENDS


# ─────────────────────────────────────────────────────────────────────────────
# Encoder wrapper (ported verbatim from serving/backends/onnx_cuda.py)
# ─────────────────────────────────────────────────────────────────────────────
class _ORTEncoderWrapper(torch.nn.Module):
    """Drop-in replacement for `model.encoder` that routes forward through ORT.

    Mimics HuggingFace's `transformers.modeling_outputs.BaseModelOutput`:
    returns an object with `.last_hidden_state`.
    """

    def __init__(self, ort_sess, hidden_size: int, device: torch.device, dtype: torch.dtype,
                 base_config=None, ort_dtype: np.dtype = np.float32,
                 use_iobinding: bool = True):
        super().__init__()
        self._ort = ort_sess
        self._input_names = {i.name for i in ort_sess.get_inputs()}
        self._output_name = ort_sess.get_outputs()[0].name
        self._device = device
        self._dtype = dtype
        self._ort_dtype = ort_dtype
        self._use_iobinding = use_iobinding
        self.hidden_size = hidden_size
        # Some upstream code reads encoder.config; pass through if available.
        if base_config is not None:
            self.config = base_config
        # Dummy parameter so `next(self.parameters())` returns something with
        # the right device/dtype (matches HF encoder expectations).
        self.register_parameter("_anchor",
                                torch.nn.Parameter(torch.zeros((), device=device, dtype=dtype),
                                                   requires_grad=False))

    @torch.inference_mode()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, **kwargs):
        from transformers.modeling_outputs import BaseModelOutput

        t0 = time.perf_counter() if _LOG_TIMING else None
        gpu = (hasattr(self._device, "type") and self._device.type == "cuda")
        B, S = (input_ids.shape if input_ids is not None else (-1, -1))

        if gpu and self._use_iobinding:
            try:
                hidden = self._run_iobinding(
                    input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                if _LOG_TIMING:
                    print(f"[onnx] forward IOBind  B={B} S={S}  "
                          f"{(time.perf_counter() - t0)*1000:.1f} ms", file=sys.stderr)
                return BaseModelOutput(last_hidden_state=hidden)
            except RuntimeError as e:
                msg = str(e)
                if "data transfer" in msg or "CUDAExecutionProvider" in msg:
                    print(f"[onnx] IOBinding failed ({msg.splitlines()[0]}); "
                          f"falling back to CPU-feed path. Active providers: "
                          f"{self._ort.get_providers()}")
                    self._use_iobinding = False
                else:
                    raise

        # CPU-feed path (CPU EP only, or IOBinding off / failed).
        feeds = {}
        if input_ids is not None and "input_ids" in self._input_names:
            feeds["input_ids"] = input_ids.detach().to(torch.int64).contiguous().cpu().numpy()
        if attention_mask is not None and "attention_mask" in self._input_names:
            feeds["attention_mask"] = attention_mask.detach().to(torch.int64).contiguous().cpu().numpy()
        if token_type_ids is not None and "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = token_type_ids.detach().to(torch.int64).contiguous().cpu().numpy()
        ort_out = self._ort.run([self._output_name], feeds)[0]
        hidden = torch.from_numpy(np.asarray(ort_out)).to(self._device).to(self._dtype)
        if _LOG_TIMING:
            print(f"[onnx] forward CPUFeed B={B} S={S}  "
                  f"{(time.perf_counter() - t0)*1000:.1f} ms", file=sys.stderr)
        return BaseModelOutput(last_hidden_state=hidden)

    def _run_iobinding(self, input_ids, attention_mask=None, token_type_ids=None) -> torch.Tensor:
        """Run the ORT session against GPU-resident tensors via IOBinding.

        Inputs are bound from PyTorch CUDA tensors (zero-copy). The output
        buffer is pre-allocated in PyTorch and bound by pointer too, so the
        encoder output never leaves GPU memory.
        """
        io = self._ort.io_binding()
        device_id = self._device.index if self._device.index is not None else 0

        pinned = []  # strong refs so input buffers aren't freed before run

        def _bind_int64(name, tensor):
            if tensor is None or name not in self._input_names:
                return
            t = tensor.detach().to(torch.int64).contiguous()
            if not t.is_cuda:
                t = t.cuda(device_id)
            io.bind_input(
                name=name, device_type="cuda", device_id=device_id,
                element_type=np.int64, shape=tuple(t.shape), buffer_ptr=t.data_ptr(),
            )
            pinned.append(t)

        _bind_int64("input_ids", input_ids)
        _bind_int64("attention_mask", attention_mask)
        _bind_int64("token_type_ids", token_type_ids)

        # Pre-allocate output in PyTorch — same dtype as the head model so
        # downstream code can consume it without an extra cast.
        B, S = pinned[0].shape[0], pinned[0].shape[1]
        torch_dtype = torch.float16 if self._dtype == torch.float16 else torch.float32
        np_dtype = np.float16 if self._dtype == torch.float16 else np.float32
        out_tensor = torch.empty((B, S, self.hidden_size), dtype=torch_dtype,
                                 device=self._device).contiguous()
        io.bind_output(
            name=self._output_name, device_type="cuda", device_id=device_id,
            element_type=np_dtype, shape=tuple(out_tensor.shape), buffer_ptr=out_tensor.data_ptr(),
        )

        self._ort.run_with_iobinding(io)
        return out_tensor


# ─────────────────────────────────────────────────────────────────────────────
# ORT session construction
# ─────────────────────────────────────────────────────────────────────────────
def _build_trt_options(onnx_path: str) -> dict:
    """TensorRT EP options (mirrors serving/backends/onnx_trt.py).

    We deliberately DO NOT set explicit shape profiles: for transformer
    encoders TRT EP partitions the graph (some attention ops fall back to
    CUDA), and the TRT subgraph inputs then include intermediate tensors we
    can't enumerate up front. Instead TRT auto-builds an engine per observed
    (batch, seq) shape and caches it — first request of each new shape pays the
    1-3 min build cost, then hits cache.
    """
    cache_dir = os.path.join(os.path.dirname(onnx_path or "."), "trt_cache")
    os.makedirs(cache_dir, exist_ok=True)
    opt_level = int(os.environ.get("SERVING_TRT_OPT_LEVEL", "1"))
    workspace_gb = int(os.environ.get("SERVING_TRT_WORKSPACE_GB", "2"))
    excluded = os.environ.get("SERVING_TRT_EXCLUDE_OPS", "Range,RandomNormalLike,ScatterND")
    print(f"[onnx_trt] engine cache: {cache_dir}  workspace={workspace_gb}GB "
          f"opt_level={opt_level}  exclude_ops={excluded}", flush=True)
    print("[onnx_trt] auto-build mode: each new (batch, seq) shape triggers a 1-3 min build "
          "on first request, then caches.", flush=True)
    return {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": cache_dir,
        "trt_max_workspace_size": workspace_gb * 1024 * 1024 * 1024,
        "trt_builder_optimization_level": opt_level,
        "trt_min_subgraph_size": 5,
        "trt_detailed_build_log": True,
        "trt_op_types_to_exclude": excluded,
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": cache_dir,
    }


def _make_session(onnx_path: str, backend: str):
    """Create an ORT InferenceSession configured for `backend`."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    if backend == "onnx_trt_fp16":
        # Preflight: fail loudly if TRT EP isn't even available — otherwise ORT
        # silently falls back to CUDA EP and the results table is misleading.
        avail = set(ort.get_available_providers())
        if "TensorrtExecutionProvider" not in avail:
            raise RuntimeError(
                "TensorrtExecutionProvider not available in this onnxruntime install.\n"
                f"Available providers: {sorted(avail)}\n"
                "Likely causes: onnxruntime-gpu wheel built without TRT, or TensorRT\n"
                "runtime libs missing. Fix: pip install tensorrt-cu12 (then re-run)."
            )
        # TRT does its own graph optimization; ORT's full pipeline can clash with
        # TRT EP partitioning (e.g. 'Failed to find kernel for Range on TRT EP').
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = [
            ("TensorrtExecutionProvider", _build_trt_options(onnx_path)),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        # The memcpy transformer is a correctness pass that still runs under
        # ORT_DISABLE_ALL; disable it so it doesn't seek TRT kernels for ops
        # TRT can't compile.
        return ort.InferenceSession(
            onnx_path, sess_options=sess_opts, providers=providers,
            disabled_optimizers=["TransformerMemcpyOptimization"],
        )

    # onnx_cuda_fp16
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, sess_options=sess_opts, providers=providers)


def _warn_if_fp32(sess, backend: str) -> None:
    """Warn if the ONNX graph is FP32 *while a GPU is present* (the real misconfig).

    On a CPU-only box FP32 is expected (no fast FP16 CPU kernels), so stay quiet.
    """
    if not torch.cuda.is_available():
        return
    in0 = sess.get_inputs()[0].type
    out0 = sess.get_outputs()[0].type
    if "float16" not in out0 and "float16" not in in0:
        print(f"[{backend}] WARNING: ONNX graph appears FP32 (out type={out0}). "
              f"Re-export with FP16 (default) for a fair quantized comparison.")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def attach_onnx_encoder(model, onnx_path: str, backend: str, *, use_iobinding=None):
    """Swap `model.encoder` for an ORT-backed wrapper. Returns the same model.

    The model's heads should already be on their target device/dtype (the
    wrapper output dtype is matched to `next(model.parameters()).dtype`). Call
    AFTER `model.to(device)` (+ optional `.half()`) so PyTorch claims the CUDA
    context before ORT — otherwise ORT-CUDA can silently fall back to slow paths.
    """
    if backend not in ONNX_BACKENDS:
        raise ValueError(f"attach_onnx_encoder: backend must be one of {ONNX_BACKENDS}, got {backend!r}")
    if not onnx_path or not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"ONNX file not found: {onnx_path!r}. Export it first "
            f"(evaluate.py --export-onnx, or models.onnx_export.export_encoder_onnx)."
        )

    sess = _make_session(onnx_path, backend)
    active = sess.get_providers()
    has_gpu = any(p.startswith(("CUDA", "Tensorrt")) for p in active)
    print(f"[{backend}] ORT active providers: {active}")
    if backend == "onnx_trt_fp16" and active and active[0] != "TensorrtExecutionProvider":
        print("[onnx_trt] WARNING: TensorrtExecutionProvider is not primary — "
              "session will fall back to CUDA/CPU; numbers won't reflect TRT.")
    if not has_gpu:
        print(f"[{backend}] WARNING: no GPU provider active (CPU-feed path, slow). "
              f"Check onnxruntime-gpu install + CUDA/cuDNN versions.")
    _warn_if_fp32(sess, backend)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    base_cfg = getattr(model.encoder, "config", None)
    hidden_size = getattr(model, "hidden_size", None) or (
        getattr(base_cfg, "hidden_size", None) if base_cfg is not None else None
    )
    if use_iobinding is None:
        use_iobinding = (os.environ.get("SERVING_IOBINDING", "0") == "1") and has_gpu
    ort_dtype = np.float16 if dtype == torch.float16 else np.float32

    wrapper = _ORTEncoderWrapper(
        ort_sess=sess, hidden_size=hidden_size or 0, device=device, dtype=dtype,
        base_config=base_cfg, ort_dtype=ort_dtype, use_iobinding=use_iobinding,
    )
    print(f"[{backend}] IOBinding: {'ENABLED' if use_iobinding else 'disabled (set SERVING_IOBINDING=1)'}")
    model.encoder = wrapper
    return model


def build_backend_model(model, backend: str, onnx_path: str = None):
    """Transform a loaded GLiNER2 model into the requested backend variant.

    pytorch_fp32 → unchanged (baseline).
    pytorch_fp16 → model.half() (CUDA only).
    onnx_*       → model.half() (CUDA only) + encoder swapped for ORT session.

    Returns the (possibly mutated) model. Encoder swap mutates in place, so
    pass a freshly-loaded model when benchmarking multiple backends.
    """
    if backend not in ALL_BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Known: {ALL_BACKENDS}")

    on_cuda = torch.cuda.is_available()

    if backend == "pytorch_fp32":
        return model

    if backend == "pytorch_fp16":
        if on_cuda:
            model = model.half()
        else:
            print("[pytorch_fp16] WARNING: CUDA unavailable; keeping FP32 "
                  "(CPU has no fast FP16 path).")
        return model

    # onnx_cuda_fp16 / onnx_trt_fp16
    if on_cuda:
        model = model.half()
    return attach_onnx_encoder(model, onnx_path, backend)
