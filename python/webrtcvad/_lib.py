from __future__ import annotations

import ctypes
import os
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "webrtcvad.mojo")
LIB = os.environ.get("MOJO_WEBRTCVAD_LIB") or os.path.join(
    ROOT, "dist", "libmojo-webrtcvad.so"
)
I = ctypes.c_int64

_SIGNATURES = {
    "mwv_state_length": ([], I),
    "mwv_init": ([I], I),
    "mwv_set_mode": ([I, I], I),
    "mwv_valid_rate_and_frame_length": ([I, I], I),
    "mwv_process": ([I, I, I, I], I),
    "mwv_process_many": ([I, I, I, I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def build() -> str:
    if os.path.exists(LIB) and (
        not os.path.exists(SRC) or os.path.getmtime(LIB) >= os.path.getmtime(SRC)
    ):
        return LIB
    script = os.path.join(ROOT, "build", "build.sh")
    proc = subprocess.run(
        ["bash", script], capture_output=True, text=True, timeout=1800
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library
