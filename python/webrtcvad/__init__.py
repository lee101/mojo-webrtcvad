from __future__ import annotations

import ctypes
import operator

import numpy as np

from ._lib import lib

__author__ = "Lee Penkman"
__copyright__ = "Copyright (C) 2026 Lee Penkman"
__license__ = "MIT"
__version__ = "0.1.0"

_bytes_address = ctypes.pythonapi.PyBytes_AsString
_bytes_address.argtypes = [ctypes.py_object]
_bytes_address.restype = ctypes.c_void_p


class Error(Exception):
    pass


def _index(value, name):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if result > 2**63 - 1 or result < -(2**63):
        raise OverflowError(f"{name} is outside the signed 64-bit range")
    return result


def _c_int(value, name):
    result = _index(value, name)
    if result > 2**31 - 1 or result < -(2**31):
        raise ValueError(f"{result} is an invalid {name}")
    return result


def _buffer_address(buf, length):
    """Return an int16-readable address while keeping the exporter alive."""
    if type(buf) is bytes:
        size = len(buf)
        owner = buf
        address = _bytes_address(owner)
    else:
        try:
            view = memoryview(buf)
        except TypeError as exc:
            raise Error("Error while processing frame") from exc
        if not view.c_contiguous:
            raise Error("PCM buffer must be C-contiguous")
        size = view.nbytes
        try:
            owner = np.frombuffer(view, dtype=np.uint8, count=size)
        except (TypeError, ValueError, BufferError) as exc:
            raise Error("Error while processing frame") from exc
        address = owner.ctypes.data
    if length * 2 > size:
        raise IndexError(
            f"buffer has {size // 2} frames, but length argument was {length}"
        )
    if not address:
        raise Error("PCM buffer has a null address")
    return address, owner, size


def valid_rate_and_frame_length(rate, frame_length):
    rate = _c_int(rate, "rate")
    frame_length = _c_int(frame_length, "frame length")
    return bool(lib().mwv_valid_rate_and_frame_length(rate, frame_length))


class Vad:
    def __init__(self, mode=None):
        kernel = lib()
        self._state = np.empty(kernel.mwv_state_length(), dtype=np.int32)
        self._state_address = self._state.ctypes.data
        self._process = kernel.mwv_process
        self._process_many = kernel.mwv_process_many
        self._set_mode = kernel.mwv_set_mode
        if not self._state_address or kernel.mwv_init(self._state_address):
            raise Error("Unable to initialize detector")
        if mode is not None:
            self.set_mode(mode)

    def set_mode(self, mode):
        mode = _c_int(mode, "mode")
        if mode < 0 or mode > 3:
            raise ValueError(f"{mode} is an invalid mode, must be 0-3")
        if self._set_mode(self._state_address, mode):
            raise Error(f"Unable to set mode to {mode}")

    def is_speech(self, buf, sample_rate, length=None):
        sample_rate = _c_int(sample_rate, "sample rate")
        if length is None:
            try:
                size = len(buf) if type(buf) is bytes else memoryview(buf).nbytes
            except TypeError as exc:
                raise Error("Error while processing frame") from exc
            length = size // 2
        else:
            length = _c_int(length, "length")
        if length < 0:
            raise ValueError("length must be non-negative")
        sample_address, owner, _ = _buffer_address(buf, length)
        result = self._process(
            self._state_address, sample_address, sample_rate, length
        )
        _ = owner
        if result < 0:
            raise Error("Error while processing frame")
        return bool(result)

    def is_speech_many(self, buf, sample_rate, frame_length):
        """Classify concatenated fixed-size PCM frames with one FFI call."""
        sample_rate = _c_int(sample_rate, "sample rate")
        frame_length = _c_int(frame_length, "frame length")
        frame_bytes = frame_length * 2
        try:
            size = len(buf) if type(buf) is bytes else memoryview(buf).nbytes
        except TypeError as exc:
            raise Error("Error while processing frames") from exc
        if frame_bytes <= 0 or size % frame_bytes:
            raise ValueError("buffer length must be a multiple of frame_length * 2")
        frame_count = size // frame_bytes
        sample_address, owner, _ = _buffer_address(buf, size // 2)
        decisions = np.empty(frame_count, dtype=np.bool_)
        result = self._process_many(
            self._state_address,
            sample_address,
            decisions.ctypes.data,
            sample_rate,
            frame_length,
            frame_count,
        )
        _ = owner
        if result < 0:
            raise Error("Error while processing frames")
        return decisions
