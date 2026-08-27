# mojo-webrtcvad

`mojo-webrtcvad` is a standalone Mojo port of the fixed-point WebRTC voice
activity detector exposed by Python's `webrtcvad` package. It implements the
actual six-band filterbank, adaptive Gaussian mixture models, sample-rate
conversion, aggressiveness thresholds, and stateful hangover logic. It is not
an energy-threshold approximation.

The Python module deliberately uses the upstream import name:

```python
import webrtcvad

vad = webrtcvad.Vad(2)
frame = bytes(320)  # 10 ms of mono 16-bit PCM at 16 kHz
print(vad.is_speech(frame, 16000))  # False
```

## Coverage

The complete public API of `webrtcvad` 2.0.10 is covered:

- `Vad(mode=None)`
- `Vad.set_mode(mode)` for modes 0 through 3
- `Vad.is_speech(buf, sample_rate, length=None)`
- `valid_rate_and_frame_length(rate, frame_length)`
- 8, 16, 32, and 48 kHz mono signed 16-bit PCM
- 10, 20, and 30 ms frames

`Vad.is_speech_many(buf, sample_rate, frame_length)` is an additional batch
API. It accepts concatenated equal-size frames, preserves detector state in
frame order, and returns a NumPy boolean array. It avoids one Python and ctypes
round trip per frame.

The private upstream `_webrtcvad` C-extension API is not reproduced. Arbitrary
sample rates, stereo audio, floating-point PCM, and frames outside 10/20/30 ms
are also out of scope, as they are upstream.

## Install and build

```bash
pixi install
pixi run build
pixi run test
```

The build task compiles the single Mojo compilation unit with
`mojo build --emit shared-lib` and writes
`dist/libmojo-webrtcvad.so`. The Pixi environment also installs upstream
`webrtcvad==2.0.10` so the test suite can compare every supported rate,
duration, and mode against the real package.

For batched audio:

```python
import webrtcvad

rate = 16000
frame_length = 160
with open("speech-16khz-mono-s16le.raw", "rb") as source:
    pcm = source.read()

decisions = webrtcvad.Vad(2).is_speech_many(pcm, rate, frame_length)
voiced_frames = decisions.nonzero()[0]
```

The input file must contain headerless, mono, little-endian signed 16-bit PCM,
and its length must be a multiple of `frame_length * 2`.

## Correctness

The tests use deterministic silence, low noise, speech-band tones, broadband
noise, clipped PCM extremes, mode changes, and state split across multiple
calls. Decisions are asserted against upstream 2.0.10 over all 48
rate/duration/mode combinations. The batch API is checked against an upstream
per-frame loop. The current suite has 78 passing tests, including explicit SIMD
remainder and zero-copy buffer-path coverage.

The implementation preserves observable 2.0.10 fixed-point and resampler
behavior, including integer wrapping, rounding, model adaptation, and
hysteresis. The Mojo source is a translation of WebRTC code distributed under
the BSD license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-136-generic. Values are best-of-five wall-clock times and include
the public Python API. Ratios below 1 mean Mojo is slower.

| case | mojo-webrtcvad | upstream | upstream / Mojo |
| --- | ---: | ---: | ---: |
| scalar 8 kHz / 10 ms | 3.39 us/frame | 1.67 us/frame | 0.49x slower |
| scalar 16 kHz / 20 ms | 4.26 us/frame | 2.87 us/frame | 0.67x slower |
| scalar 48 kHz / 30 ms | 18.58 us/frame | 22.09 us/frame | 1.19x faster |
| batch 30 s, 16 kHz / 10 ms | 2.04 us/frame | 2.35 us/frame | 1.15x faster |
| batch 30 s, 48 kHz / 30 ms | 17.20 us/frame | 21.67 us/frame | 1.26x faster |

Upstream's scalar API is a direct CPython C extension and is faster for the
small 8 and 16 kHz frames in this run. The exact-bytes fast path passes the
CPython buffer address directly, skips generic buffer helpers, and caches
ctypes functions and the detector-state address. Other contiguous
buffer-protocol inputs remain zero-copy through NumPy. Batching removes most
boundary cost and is 15–26 percent faster than upstream in this run.

The independent filter combination, energy reduction, PCM widening, and
48-to-32 kHz FIR use native-width integer SIMD with scalar remainder loops.
The 8 kHz path widens PCM directly into the filterbank input, while the 16 kHz
decimator reads the int16 PCM buffer directly instead of materializing an
int32 copy. Independent all-pass branches are fused and inlined; recursive
state stays in registers between samples. Batch frames are not
thread-parallelized because both the adaptive detector and resampler carry
state into the next frame; the only independent FIR section has just 80 output
blocks per 10 ms chunk, below a useful thread-launch threshold.

No GPU path is provided. Even the densest FIR performs well under two fixed-point
operations per byte moved, while the other hot filters are recursive and
stateful. Device transfer and launch overhead would dominate these 10–30 ms
frames, so a GPU path would lose rather than provide a justified acceleration.

## How it works

Python owns the audio buffer and a 16 KiB contiguous `int32` detector workspace.
The workspace contains persistent filter/model state followed by reusable
scratch regions; the Mojo library performs no allocation. C ABI exports take
buffer addresses as 64-bit integers, reconstruct
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside Mojo, and return only
status or speech decisions.

PCM remains contiguous signed `int16` throughout the FFI. Filterbank samples,
Gaussian parameters, and resampler state use WebRTC's original fixed-point
Q-formats. Scalar calls classify one frame. The batch export advances through
a contiguous PCM buffer inside Mojo and writes one byte per decision before
returning to Python.

## License

Repository code is MIT licensed. The translated WebRTC VAD algorithm retains
its BSD notice in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
