"""Export a loaded GLiNER2 model's encoder to ONNX.

Adapted from the `serving-bench` branch (`serving/export/to_onnx.py`). The
difference: this takes an already-loaded GLiNER2 / CustomExtractor model
(from `models.loader.load_gliner2_model`) instead of the serving `ServingModel`
registry, so it slots directly into `evaluation/evaluate.py`.

Strategy: **encoder-only**. The multi-head GLiNER2 graph (SpanRep + CountLSTM
+ ...) is painful in ONNX, but the transformer encoder is >90% of the latency,
so exporting just the encoder captures the speedup. The runtime backend
(`models.onnx_backend`) then wraps the ONNX session as a drop-in replacement
for `model.encoder`, leaving `model.extract()` untouched.

Output file:
    <out_dir>/<model_slug>_encoder.onnx
"""
import os

import torch


def _propagate_shapes(onnx_path: str) -> None:
    """Run shape inference on the ONNX graph in-place.

    ORT's TensorRT EP refuses to compile a subgraph whose intermediate tensors
    don't carry shapes (it errors with 'TensorRT input: <X> has no shape
    specified'). torch.onnx.export with the legacy (TorchScript-trace) path
    annotates inputs/outputs but often leaves internal tensors un-shaped.

    Prefer ORT's symbolic shape inference because it handles dynamic
    (string-named) axes; fall back to plain `onnx.shape_inference` if the
    helper isn't importable. CUDA EP doesn't need this — only TRT EP does.
    """
    import onnx
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        inferred = SymbolicShapeInference.infer_shapes(
            onnx.load(onnx_path), auto_merge=True, int_max=2**31 - 1,
        )
        if inferred is not None:
            onnx.save(inferred, onnx_path)
            print("[onnx_export] symbolic shape inference -> applied")
            return
    except Exception as e:
        print(f"[onnx_export] symbolic shape inference unavailable ({type(e).__name__}: {e}); "
              f"falling back to onnx.shape_inference")
    try:
        from onnx import shape_inference
        onnx.save(shape_inference.infer_shapes(onnx.load(onnx_path)), onnx_path)
        print("[onnx_export] basic shape inference -> applied")
    except Exception as e:
        print(f"[onnx_export] WARNING: shape inference failed ({type(e).__name__}: {e}). "
              f"TRT EP will likely refuse this file.")


def _resolve_encoder_and_tokenizer(model):
    """Locate the HF encoder module + a callable HF tokenizer on a GLiNER2 model.

    Works for both mDeBERTa-v3 (GLiNER2) and mmBERT (CustomExtractor) — both expose
    the transformer at `model.encoder` and the tokenizer at
    `model.processor.tokenizer`.
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise RuntimeError(f"Could not locate encoder on {type(model).__name__}")

    # GLiNER2 stores the HF tokenizer under processor.tokenizer; the
    # `processor` attribute itself is a SchemaTransformer (non-callable).
    tok = getattr(getattr(model, "processor", None), "tokenizer", None) \
        or getattr(model, "tokenizer", None)
    if tok is None or not callable(tok):
        raise RuntimeError("Could not locate a callable HF tokenizer on the loaded model.")
    return encoder, tok


def export_encoder_onnx(model, out_path: str, fp16: bool = True, opset: int = 17) -> str:
    """Export `model.encoder` to ONNX at `out_path`.

    Args:
        model:    a loaded GLiNER2 / CustomExtractor model.
        out_path: destination .onnx path (parent dirs created).
        fp16:     export in FP16 (default; required for a fair quantized
                  comparison). Only applied on CUDA — on CPU the export stays
                  FP32 since most CPU ONNX ops have no FP16 kernel.
        opset:    ONNX opset version.

    Returns the written path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    encoder, tok = _resolve_encoder_and_tokenizer(model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = fp16 and device.startswith("cuda")

    encoder = encoder.to(device).eval()
    if use_fp16:
        encoder = encoder.half()

    sample = tok(["Hello world, my email is foo@bar.com"], return_tensors="pt",
                 padding=True, truncation=True, max_length=256)
    sample = {k: (v.to(device).half() if use_fp16 and v.dtype == torch.float32 else v.to(device))
              for k, v in sample.items()}

    dyn_axes = {
        "input_ids":      {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
    }
    input_names = list(sample.keys())
    with torch.inference_mode():
        torch.onnx.export(
            encoder, tuple(sample.values()), out_path,
            input_names=input_names, output_names=["hidden_states"],
            dynamic_axes={**{k: v for k, v in dyn_axes.items() if k in input_names},
                          "hidden_states": {0: "batch", 1: "seq"}},
            opset_version=opset, do_constant_folding=True, dynamo=False,
        )
    _propagate_shapes(out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[onnx_export] encoder -> {out_path} ({size_mb:.1f} MB, fp16={use_fp16})")
    return out_path
