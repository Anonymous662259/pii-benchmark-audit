"""Wire pip-installed CUDA 12 / TRT libs into the dynamic loader path.

Ported from the `serving-bench` branch (`serving/_cuda_libs.py`) so the
ONNX-CUDA / TensorRT eval backends can load on Modal.

Why this exists
---------------
Modal's default images ship CUDA 13 (PyTorch +cu130) but the
`onnxruntime-gpu` 1.20+ wheels expect CUDA 12 (`libcublasLt.so.12`,
`libcudnn.so.9`, `libcudart.so.12`, ...). The lib version names differ
across major CUDA releases, so ORT cannot satisfy itself from the system
CUDA 13 install. The conventional fix is to pip-install NVIDIA's CUDA 12
runtime wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`,
`nvidia-cuda-runtime-cu12`, ...) alongside, but those wheels drop their
.so files under `site-packages/nvidia/<lib>/lib/` which is NOT on the
default `LD_LIBRARY_PATH`. So ORT still can't dlopen them.

This module finds those paths and:

1) Updates `os.environ['LD_LIBRARY_PATH']` so any subprocess we spawn
   inherits the right loader path.
2) Best-effort `ctypes.CDLL(..., RTLD_GLOBAL)` to make the symbols
   available inside the *current* Python process, where LD_LIBRARY_PATH
   changes come too late (libc reads it at process startup only).

The module auto-runs on import. Disable with `SERVING_SKIP_CUDA_PRELOAD=1`.
Set `SERVING_VERBOSE_PRELOAD=1` to print what it found. (Env-var names kept
identical to the serving-bench branch so docs / muscle memory carry over.)

On non-Linux (e.g. Windows dev box) there are no `.so` wheels to find, so
this is a no-op — safe to import everywhere.
"""
import ctypes
import glob
import os
import sys
from typing import List


def _find_lib_dirs() -> List[str]:
    """Return the lib directories shipped by pip-installed CUDA / TRT wheels."""
    dirs: List[str] = []
    seen = set()

    site_packages = [p for p in sys.path if p.rstrip("/").endswith("site-packages")]
    for sp in site_packages:
        # nvidia-* wheels: site-packages/nvidia/<lib>/lib/*.so*
        for d in sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib"))):
            if d not in seen and os.path.isdir(d):
                dirs.append(d); seen.add(d)
        # TensorRT wheels: site-packages/tensorrt_libs/*.so* (cu12 or cu13)
        for d in (os.path.join(sp, "tensorrt_libs"),
                  os.path.join(sp, "tensorrt_cu12_libs"),
                  os.path.join(sp, "tensorrt_cu13_libs")):
            if d not in seen and os.path.isdir(d):
                dirs.append(d); seen.add(d)
    return dirs


def preload_cuda_libs(verbose: bool = False) -> dict:
    """Make pip-installed CUDA / TRT libs visible to ORT.

    Returns:
        dict with `dirs`, `preloaded`, `subprocess_env_set` so callers can
        report what happened.
    """
    dirs = _find_lib_dirs()
    info = {"dirs": dirs, "preloaded": 0, "subprocess_env_set": False}
    if not dirs:
        if verbose:
            print("[cuda_libs] no pip-installed nvidia/tensorrt wheels found")
        return info

    # (1) subprocess inheritance
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    cur_parts = cur.split(":") if cur else []
    new_parts = [d for d in dirs if d not in cur_parts]
    if new_parts:
        os.environ["LD_LIBRARY_PATH"] = ":".join(new_parts + cur_parts)
        info["subprocess_env_set"] = True

    # (2) ctypes preload — every .so* file in those dirs. We attempt all,
    # ignoring failures (some libs are symlinks or already loaded).
    for d in dirs:
        for lib in sorted(glob.glob(os.path.join(d, "*.so*"))):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                info["preloaded"] += 1
            except OSError:
                pass

    if verbose:
        print(f"[cuda_libs] added {len(dirs)} dirs, preloaded {info['preloaded']} libs")
        for d in dirs:
            print(f"           {d}")
    return info


# Auto-run on import unless disabled.
if os.environ.get("SERVING_SKIP_CUDA_PRELOAD", "0") != "1":
    _PRELOAD_INFO = preload_cuda_libs(
        verbose=os.environ.get("SERVING_VERBOSE_PRELOAD", "0") == "1"
    )
else:
    _PRELOAD_INFO = {"dirs": [], "preloaded": 0, "subprocess_env_set": False}
