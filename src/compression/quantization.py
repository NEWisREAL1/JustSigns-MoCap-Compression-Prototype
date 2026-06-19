import base64
from copy import deepcopy

import numpy as np


class QuantizationCompressor:
    """
    Compressing clips using simple quantization. Bypassing the 'ASCII Tax'
    by encoding the quantized integer arrays directly into Base64 strings.
    """

    def __init__(self, base=np.uint8, renormalize=True):
        self.base_type = base
        self.num_bits = np.dtype(base).itemsize * 8
        self.renormalize = renormalize

    # ----- Public API ----- #

    def compress(self, clip):
        tracks = clip["animationClip"]["tracks"]
        compressed_tracks = []

        for track in tracks:
            times = np.asarray(track["times"])
            if track["type"] == "quaternion":
                values = np.asarray(track["values"]).reshape((-1, 4))
            else:
                values = np.asarray(track["values"])

            q_times, t_scale, t_zero = self._quantize(times)
            q_values, v_scale, v_zero = self._quantize(values)

            # append '_b64' so your Three.js decoder knows to unpack it
            # .tolist() ensures numpy floats are converted to native Python floats for JSON
            com_track = {
                "name": track["name"],
                "type": track["type"] + "_b64", 
                "times": {
                    "codes_b64": self._pack_b64(q_times),
                    "scale": np.asarray(t_scale).tolist(), 
                    "zero" : np.asarray(t_zero).tolist(),
                },
                "values": {
                    "codes_b64": self._pack_b64(q_values),
                    "scale": np.asarray(v_scale).tolist(),
                    "zero" : np.asarray(v_zero).tolist(),
                },
            }

            compressed_tracks.append(com_track)

        compressed_clip = deepcopy(clip)
        compressed_clip["animationClip"]["tracks"] = compressed_tracks
        return compressed_clip

    def decompress(self, compressed_clip):
        compressed_tracks = compressed_clip["animationClip"]["tracks"]
        decompressed_tracks = []

        for track in compressed_tracks:
            times = track["times"]
            values = track["values"]
            
            # revert the type name for standard processing downstream
            original_type = track["type"].replace("_b64", "")

            # decode Base64 strings back to flat 1D numpy integer arrays
            q_times = self._unpack_b64(times["codes_b64"])
            q_values = self._unpack_b64(values["codes_b64"])

            # dequantize times
            dequantized_times = self._dequantize(q_times, times["scale"], times["zero"])
            
            # dequantize values
            if original_type == "quaternion":
                q_values = q_values.reshape((-1, 4))
                dequantized_values = self._dequantize(q_values, values["scale"], values["zero"])

                if self.renormalize:
                    norms = np.linalg.norm(dequantized_values, axis=1, keepdims=True)
                    # Add epsilon to prevent div by zero
                    dequantized_values = dequantized_values / (norms + 1e-8)
            else:
                dequantized_values = self._dequantize(q_values, values["scale"], values["zero"])

            decom_track = {
                "name"  : track["name"],
                "type"  : original_type,
                "times" : dequantized_times.tolist(),
                "values": dequantized_values.reshape(-1).tolist(),
            }

            decompressed_tracks.append(decom_track)

        decompressed_clip = deepcopy(compressed_clip)
        decompressed_clip["animationClip"]["tracks"] = decompressed_tracks
        return decompressed_clip


    # ----- Helper API ----- #

    def _pack_b64(self, arr):
        """Converts numpy array to bytes, then to base64 ASCII string"""
        return base64.b64encode(arr.tobytes()).decode('ascii')

    def _unpack_b64(self, b64_str):
        """Converts base64 ASCII string back to numpy array of the correct base type"""
        return np.frombuffer(base64.b64decode(b64_str), dtype=self.base_type)

    def _quantize(self, data):
        val_min, val_max = np.min(data, axis=0), np.max(data, axis=0)

        # Handle flat lines (zero variance)
        if np.all(np.isclose(val_min, val_max)):
            scale_factor = np.ones_like(val_max, dtype=float)
            zero_point = np.zeros_like(val_max, dtype=float)
            return np.zeros_like(data, dtype=self.base_type), scale_factor, zero_point

        scale_factor = (val_max - val_min) / (2 ** self.num_bits - 1)
        zero_point = np.round(-val_min / scale_factor)

        q_data = np.round(data / scale_factor) + zero_point
        q_data = q_data.astype(self.base_type)

        return q_data, scale_factor, zero_point

    def _dequantize(self, q_data, scale_factor, zero_point):
        # Cast scale/zero back to numpy arrays for vectorized math
        sf = np.asarray(scale_factor)
        zp = np.asarray(zero_point)
        return sf * (q_data - zp)