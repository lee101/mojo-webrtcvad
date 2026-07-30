"""Benchmarks mojo-webrtcvad against upstream webrtcvad 2.0.10."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import platform
import site
import sys
import time
import types

import numpy as np

import webrtcvad


def load_upstream():
    package_resources = types.ModuleType("pkg_resources")
    package_resources.get_distribution = lambda _: types.SimpleNamespace(
        version="2.0.10"
    )
    sys.modules.setdefault("pkg_resources", package_resources)
    path = next(
        pathlib.Path(directory, "webrtcvad.py")
        for directory in site.getsitepackages()
        if pathlib.Path(directory, "webrtcvad.py").exists()
    )
    spec = importlib.util.spec_from_file_location("upstream_webrtcvad", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


UPSTREAM = load_upstream()


def pcm(rate: int, duration_ms: int, frames: int, seed: int = 7) -> bytes:
    rng = np.random.default_rng(seed)
    length = rate * duration_ms // 1000
    time_axis = np.arange(length) / rate
    chunks = []
    for index in range(frames):
        if index % 5 == 0:
            values = rng.normal(0, 30, length)
        elif index % 5 in (1, 2, 3):
            values = 9000 * np.sin(
                2 * np.pi * (180 + index % 13 * 23) * time_axis
            )
            values += rng.normal(0, 300, length)
        else:
            values = rng.normal(0, 5000, length)
        chunks.append(
            np.clip(np.rint(values), -32768, 32767).astype("<i2").tobytes()
        )
    return b"".join(chunks)


def best_time(function, repeat: int = 5) -> float:
    function()
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def scalar_case(rate: int, duration_ms: int, calls: int):
    length = rate * duration_ms // 1000
    frame = pcm(rate, duration_ms, 1)
    ours = webrtcvad.Vad(2)
    theirs = UPSTREAM.Vad(2)

    def mojo_call():
        for _ in range(calls):
            ours.is_speech(frame, rate)

    def upstream_call():
        for _ in range(calls):
            theirs.is_speech(frame, rate)

    return mojo_call, upstream_call, calls


def batch_case(rate: int, duration_ms: int, frames: int):
    length = rate * duration_ms // 1000
    audio = pcm(rate, duration_ms, frames)
    check_ours = webrtcvad.Vad(2).is_speech_many(audio, rate, length)
    check_upstream = UPSTREAM.Vad(2)
    chunks = [
        audio[offset : offset + length * 2]
        for offset in range(0, len(audio), length * 2)
    ]
    check_reference = np.array(
        [check_upstream.is_speech(frame, rate) for frame in chunks], dtype=bool
    )
    if not np.array_equal(check_ours, check_reference):
        raise RuntimeError("benchmark inputs do not produce parity")
    ours = webrtcvad.Vad(2)
    theirs = UPSTREAM.Vad(2)

    def mojo_call():
        ours.is_speech_many(audio, rate, length)

    def upstream_call():
        [theirs.is_speech(frame, rate) for frame in chunks]

    return mojo_call, upstream_call, frames


def cpu_name() -> str:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    cases = [
        ("scalar 8 kHz / 10 ms", scalar_case(8000, 10, 20_000)),
        ("scalar 16 kHz / 20 ms", scalar_case(16000, 20, 15_000)),
        ("scalar 48 kHz / 30 ms", scalar_case(48000, 30, 8_000)),
        ("batch 30 s, 16 kHz / 10 ms", batch_case(16000, 10, 3_000)),
        ("batch 30 s, 48 kHz / 30 ms", batch_case(48000, 30, 1_000)),
    ]
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| case | mojo-webrtcvad | upstream | upstream / Mojo |")
    print("| --- | ---: | ---: | ---: |")
    for name, (ours, theirs, units) in cases:
        mojo_seconds = best_time(ours)
        upstream_seconds = best_time(theirs)
        mojo_us = mojo_seconds * 1e6 / units
        upstream_us = upstream_seconds * 1e6 / units
        ratio = upstream_seconds / mojo_seconds
        suffix = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_us:.2f} us/frame | "
            f"{upstream_us:.2f} us/frame | {ratio:.2f}x {suffix} |"
        )


if __name__ == "__main__":
    main()
