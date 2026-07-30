from __future__ import annotations

import numpy as np
import pytest

import webrtcvad


RATES = (8000, 16000, 32000, 48000)
DURATIONS = (10, 20, 30)
MODES = (0, 1, 2, 3)


def pcm_stream(rate: int, duration_ms: int, frames: int = 72) -> list[bytes]:
    length = rate * duration_ms // 1000
    rng = np.random.default_rng(rate + duration_ms)
    time = np.arange(length) / rate
    result = []
    for index in range(frames):
        phase = index % 6
        if phase == 0:
            samples = np.zeros(length)
        elif phase == 1:
            samples = rng.normal(0, 24, length)
        elif phase in (2, 3):
            frequency = 180 + 37 * (index % 9)
            samples = 9000 * np.sin(2 * np.pi * frequency * time)
            samples += rng.normal(0, 250, length)
        elif phase == 4:
            samples = rng.normal(0, 3500, length)
        else:
            samples = rng.normal(0, 15000, length)
        result.append(
            np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()
        )
    return result


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("duration_ms", DURATIONS)
@pytest.mark.parametrize("mode", MODES)
def test_stateful_decisions_match_upstream(
    upstream_webrtcvad, rate, duration_ms, mode
):
    ours = webrtcvad.Vad(mode)
    theirs = upstream_webrtcvad.Vad(mode)
    frames = pcm_stream(rate, duration_ms)
    assert [ours.is_speech(frame, rate) for frame in frames] == [
        theirs.is_speech(frame, rate) for frame in frames
    ]


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("duration_ms", DURATIONS)
def test_batch_matches_upstream(upstream_webrtcvad, rate, duration_ms):
    frames = pcm_stream(rate, duration_ms, frames=40)
    length = rate * duration_ms // 1000
    ours = webrtcvad.Vad(2).is_speech_many(b"".join(frames), rate, length)
    reference = upstream_webrtcvad.Vad(2)
    expected = [reference.is_speech(frame, rate) for frame in frames]
    assert ours.dtype == np.bool_
    assert ours.tolist() == expected


def test_batch_preserves_state_across_calls():
    frames = pcm_stream(16000, 10, frames=60)
    joined = b"".join(frames)
    whole = webrtcvad.Vad(1).is_speech_many(joined, 16000, 160)
    split_detector = webrtcvad.Vad(1)
    split = np.r_[
        split_detector.is_speech_many(b"".join(frames[:23]), 16000, 160),
        split_detector.is_speech_many(b"".join(frames[23:]), 16000, 160),
    ]
    assert np.array_equal(whole, split)


def test_mode_change_matches_upstream(upstream_webrtcvad):
    frames = pcm_stream(8000, 30)
    ours = webrtcvad.Vad(0)
    theirs = upstream_webrtcvad.Vad(0)
    got = []
    expected = []
    for index, frame in enumerate(frames):
        if index == 29:
            ours.set_mode(3)
            theirs.set_mode(3)
        got.append(ours.is_speech(frame, 8000))
        expected.append(theirs.is_speech(frame, 8000))
    assert got == expected


def test_pcm_extremes_match_upstream(upstream_webrtcvad):
    samples = np.resize(np.array([-32768, 32767, 0, 32767], dtype="<i2"), 1440)
    frame = samples.tobytes()
    for mode in MODES:
        assert webrtcvad.Vad(mode).is_speech(frame, 48000) == (
            upstream_webrtcvad.Vad(mode).is_speech(frame, 48000)
        )


def test_simd_tail_paths_match_upstream(upstream_webrtcvad):
    rng = np.random.default_rng(41)
    frames = []
    for index in range(25):
        samples = rng.integers(-32768, 32768, 80, dtype=np.int16)
        samples[index % 80] = -32768
        samples[(index * 3 + 1) % 80] = 32767
        frames.append(samples.astype("<i2", copy=False).tobytes())
    ours = webrtcvad.Vad(3)
    theirs = upstream_webrtcvad.Vad(3)
    assert [ours.is_speech(frame, 8000) for frame in frames] == [
        theirs.is_speech(frame, 8000) for frame in frames
    ]


def test_zero_copy_buffer_routes_match(upstream_webrtcvad):
    frame = pcm_stream(16000, 20, frames=1)[0]
    for buffer in (frame, bytearray(frame), memoryview(frame)):
        assert webrtcvad.Vad(2).is_speech(buffer, 16000) == (
            upstream_webrtcvad.Vad(2).is_speech(buffer, 16000)
        )


def test_zero_frame_is_not_speech():
    assert not webrtcvad.Vad().is_speech(bytes(320), 16000)


def test_valid_rate_and_frame_length_matches_upstream(upstream_webrtcvad):
    for rate in (*RATES, 44100, 0, -8000):
        for length in (80, 160, 240, 320, 480, 640, 960, 1440):
            assert webrtcvad.valid_rate_and_frame_length(rate, length) == (
                upstream_webrtcvad.valid_rate_and_frame_length(rate, length)
            )


def test_invalid_mode():
    for mode in (-1, 4):
        with pytest.raises(ValueError):
            webrtcvad.Vad(mode)


def test_invalid_frame_raises_error():
    with pytest.raises(webrtcvad.Error, match="Error while processing frame"):
        webrtcvad.Vad().is_speech(bytes(200), 16000)


def test_length_guard_matches_upstream(upstream_webrtcvad):
    frame = bytes(320)
    with pytest.raises(IndexError):
        webrtcvad.Vad().is_speech(frame, 16000, 161)
    with pytest.raises(IndexError):
        upstream_webrtcvad.Vad().is_speech(frame, 16000, 161)


def test_explicit_length_uses_prefix(upstream_webrtcvad):
    rng = np.random.default_rng(9)
    samples = rng.integers(-32768, 32768, 400, dtype=np.int16).tobytes()
    assert webrtcvad.Vad(3).is_speech(samples, 8000, 80) == (
        upstream_webrtcvad.Vad(3).is_speech(samples, 8000, 80)
    )


def test_large_integer_arguments_raise():
    with pytest.raises(ValueError):
        webrtcvad.valid_rate_and_frame_length(2**35, 10)
    with pytest.raises(ValueError):
        webrtcvad.valid_rate_and_frame_length(8000, 2**35)


def test_bad_batch_shape_raises():
    with pytest.raises(ValueError, match="multiple"):
        webrtcvad.Vad().is_speech_many(bytes(321), 16000, 160)


def test_numpy_buffer_uses_byte_size(upstream_webrtcvad):
    samples = np.frombuffer(pcm_stream(16000, 10, frames=1)[0], dtype="<i2")
    assert webrtcvad.Vad(2).is_speech(samples, 16000) == (
        upstream_webrtcvad.Vad(2).is_speech(samples.tobytes(), 16000)
    )


def test_noncontiguous_buffer_is_rejected():
    samples = np.zeros(320, dtype=np.int16)[::2]
    with pytest.raises(webrtcvad.Error, match="C-contiguous"):
        webrtcvad.Vad().is_speech(samples, 16000)
    with pytest.raises(webrtcvad.Error, match="C-contiguous"):
        webrtcvad.Vad().is_speech_many(samples, 16000, 160)


@pytest.mark.parametrize("argument", [16000.0, "16000"])
def test_numeric_arguments_are_not_silently_narrowed(argument):
    with pytest.raises(TypeError):
        webrtcvad.Vad().is_speech(bytes(320), argument)
    with pytest.raises(TypeError):
        webrtcvad.valid_rate_and_frame_length(argument, 160)


def test_null_addresses_are_rejected_by_abi():
    kernel = webrtcvad._lib.lib()
    assert kernel.mwv_init(0) == -1
    assert kernel.mwv_set_mode(0, 0) == -1
    assert kernel.mwv_process(0, 0, 16000, 160) == -1
    assert kernel.mwv_process_many(0, 0, 0, 16000, 160, 1) == -1
