import base64

import numpy as np


def quantize(data, q_type=np.uint8):
    """Perform quantization on a linear list of data"""
    data_min, data_max = np.min(data), np.max(data)

    if data_min == data_max:
        return np.ones_like(data), 0, data_max

    num_bits = np.dtype(q_type).itemsize * 8
    scale = (data_max - data_min) / (2 ** num_bits - 1)
    zero  = np.round(-data_min / scale)
    codes = np.round(data / scale) + zero

    return codes.astype(q_type), scale, zero


def dequantize(codes, scale, zero):
    """Dequantize a linear list of codes back to list of continuous values"""
    return scale * (codes - zero)


def pack_b64(arr):
    """Converts numpy array to bytes, then to base64 ASCII string"""
    return base64.b64encode(arr.tobytes()).decode('ascii')


def unpack_b64(b64_str, base_type):
    """Converts base64 ASCII string back to numpy array of the base type"""
    return np.frombuffer(base64.b64decode(b64_str), dtype=base_type)