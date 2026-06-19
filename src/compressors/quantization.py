import numpy as np

from src.compressors.base import TrackCompressor
from src.data import pack_b64, unpack_b64


class QuantizeCompressor(TrackCompressor):
    
    def __init__(self, base_type=np.uint8):
        self.base_type = base_type
        self.num_bits = np.dtype(base_type).itemsize * 8

    def compress(self, track):
        # input digestion
        times  = np.asarray(track["times"] , dtype=np.float64)
        values = np.asarray(track["values"], dtype=np.float64)
        if track["type"] == "quaternion":
            values = values.reshape(-1, 4)

        # quantizations
        t_codes, t_scale, t_zero = self._quantize(times)
        v_codes, v_scale, v_zero = self._quantize(values)

        # pack track
        return {
            "name": track["name"],
            "type": track["type"] + "_quantize_b64",
            "times": {
                "codes_b64": pack_b64(t_codes),
                "scale"    : t_scale,
                "zero"     : t_zero,
            },
            "values": {
                "codes_b64": pack_b64(v_codes),
                "scale"    : v_scale,
                "zero"     : v_zero,
            },
        }

    def decompress(self, compressed_track):
        # input digestion
        times  = compressed_track["times"]
        values = compressed_track["values"]
        original_type = compressed_track["type"].replace("_quantize_b64", "")

        # Base64 unpacking
        times_codes  = unpack_b64(times["codes_b64"], self.base_type)
        values_codes = unpack_b64(values["codes_b64"], self.base_type)
        if original_type == "quaternion":
            values_codes = values_codes.reshape(-1, 4)

        # dequantizations
        dq_times  = self._dequantize(times_codes, times["scale"], times["zero"])
        dq_values = self._dequantize(values_codes, values["scale"], values["zero"])
        
        # re-normalization
        if original_type == "quaternion":
            norms = np.linalg.norm(dq_values, axis=1, keepdims=True)
            dq_values = dq_values / norms

        # pack track
        return {
            "name"   : compressed_track["name"],
            "type"   : original_type,
            "times"  : dq_times.tolist(),
            "values" : dq_values.reshape(-1).tolist(),
        }


    # ----- Helper API ----- #

    def _quantize(self, data):
        val_min, val_max = np.min(data, axis=0), np.max(data, axis=0)

        # Handle flat lines (zero variance)
        if np.all(np.isclose(val_min, val_max)):
            scale = np.ones_like(val_max, dtype=float)
            zero = np.zeros_like(val_max, dtype=float)
            return np.zeros_like(data, dtype=self.base_type), scale, zero

        scale = (val_max - val_min) / (2 ** self.num_bits - 1)
        zero  = np.round(-val_min / scale)
        codes = np.round(data / scale) + zero
        codes = codes.astype(self.base_type)
        return codes, scale, zero

    def _dequantize(self, codes, scale, zero):
        # Cast scale/zero back to numpy arrays for vectorized math
        sf = np.asarray(scale)
        zp = np.asarray(zero)
        return sf * (codes - zp)
