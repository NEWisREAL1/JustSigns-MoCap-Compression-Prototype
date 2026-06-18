import json
from copy import deepcopy

import numpy as np


class QuantizationCompressor:
    """
    Compressing clips using simple quantization.
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
            times = track["times"]
            if track["type"] == "quaternion":
                values = np.reshape(track["values"], shape=(-1, 4))
            else:
                values = track["values"]


            quantized_times, times_scale_factor, times_zero_point = self._quantize(times)
            quantized_values, values_scale_factor, values_zero_point = self._quantize(values)

            com_track = {
                "name": track["name"],
                "type": track["type"],
                "times": {
                    "codes": quantized_times,
                    "scale": times_scale_factor,
                    "zero" : times_zero_point,
                },
                "values": {
                    "codes": quantized_values.reshape((-1)),
                    "scale": values_scale_factor.reshape((-1)),
                    "zero" : values_zero_point.reshape((-1)),
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

            dequantized_times = self._dequantize(
                times["codes"], 
                times["scale"], 
                times["zero"],
                )
            
            if track["type"] == "quaternion":
                dequantized_values = self._dequantize(
                    values["codes"].reshape((-1, 4)), 
                    values["scale"].reshape((-1, 4)), 
                    values["zero"].reshape((-1, 4)),
                    )

                if self.renormalize:
                    norms = np.linalg.norm(dequantized_values, axis=1, keepdims=True)
                    dequantized_values = dequantized_values / norms
            
            else:
                dequantized_values = self._dequantize(
                    values["codes"], 
                    values["scale"], 
                    values["zero"],
                    )


            decom_track = {
                "name"  : track["name"],
                "type"  : track["type"],
                "times" : dequantized_times,
                "values": dequantized_values.reshape(-1),
            }

            decompressed_tracks.append(decom_track)

        decompressed_clip = deepcopy(compressed_clip)
        decompressed_clip["animationClip"]["tracks"] = decompressed_tracks
        return decompressed_clip


    # ----- Helper API ----- #

    def _quantize(self, data):
        val_min, val_max = np.min(data, axis=0), np.max(data, axis=0)

        if np.all(np.isclose(val_min, val_max)):
            return np.ones_like(data), val_max, np.zeros_like(val_max)

        scale_factor = (val_max - val_min) / (2 ** self.num_bits - 1)
        zero_point = np.round(-val_min / scale_factor)
        # scale_factor, zero_point = scale_factor.astype(float), zero_point.astype(float)

        q_data = np.round(data / scale_factor) + zero_point
        q_data = q_data.astype(self.base_type)

        return q_data, scale_factor, zero_point


    def _dequantize(self, q_data, scale_factor, zero_point):
        return scale_factor * (q_data - zero_point)
