import base64

import numpy as np

LEFT_HAND = [
    "pinky1_basel", "ring1_basel", "middle1_basel", "index1_basel",       
    "thumb1l", "pinky1l", "ring1l", "middle1l", "index1l",            
    "thumb2l", "pinky2l", "ring2l", "middle2l", "index2l",            
    "thumb3l", "pinky3l", "ring3l", "middle3l", "index3l",            
]

RIGHT_HAND = [
    "pinky1_baser", "ring1_baser", "middle1_baser", "index1_baser",       
    "thumb1r", "pinky1r", "ring1r", "middle1r", "index1r",            
    "thumb2r", "pinky2r", "ring2r", "middle2r", "index2r",            
    "thumb3r", "pinky3r", "ring3r", "middle3r", "index3r",            
]


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