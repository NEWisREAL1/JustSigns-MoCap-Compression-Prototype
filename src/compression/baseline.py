from copy import deepcopy

import numpy as np

from src.compression.utils import dequantize, pack_b64, quantize, unpack_b64


class RawBase64Compressor:
    
    def compress(self, clip, pass_tracks=False) -> list:
        if pass_tracks:
            tracks = clip
        else:
            tracks = clip["animationClip"]["tracks"]
        compressed_tracks = []

        for track in tracks:
            com_track = {
                "name"   : track["name"],
                "type"   : track["type"] + "_raw_b64",
                "times"  : pack_b64(np.asarray(track["times"], dtype=np.float64)),
                "values" : pack_b64(np.asarray(track["values"], dtype=np.float64)),
            }
            compressed_tracks.append(com_track)

        if pass_tracks:
            return compressed_tracks
        
        else:
            compressed_clip = deepcopy(clip)
            compressed_clip["animationClip"]["tracks"] = compressed_tracks
            return compressed_clip


    def decompress(self, compressed_clip) -> list:
        tracks = compressed_clip["animationClip"]["tracks"]
        decompressed_tracks = []

        for track in tracks:
            decom_track = {
                "name"   : track["name"],
                "type"   : track["type"].replace("_raw_b64", ""),
                "times"  : unpack_b64(track["times"], np.float64).tolist(),
                "values" : unpack_b64(track["values"], np.float64).tolist(),
            }
            decompressed_tracks.append(decom_track)

        decompressed_clip = deepcopy(compressed_clip)
        decompressed_clip["animationClip"]["tracks"] = decompressed_tracks

        return decompressed_clip


class QuantizeCompressor:
    
    def __init__(self, base_type=np.uint8):
        self.base_type = base_type

    def compress(self, clip) -> list:
        tracks = clip["animationClip"]["tracks"]
        compressed_tracks = []

        for track in tracks:
            # input digestion
            times  = np.asarray(track["times"] , dtype=np.float64)
            values = np.asarray(track["values"], dtype=np.float64)

            if track["type"] == "quaternion":
                values = values.reshape(-1, 4)

            # quantizations
            t_codes, t_scale, t_zero = quantize(times, q_type=self.base_type)
            v_codes, v_scale, v_zero = quantize(values, q_type=self.base_type)

            # pack track
            com_track = {
                "name": track["name"],
                "type": track["type"] + "_quantize_b64",
                "q_times": {
                    "codes" : pack_b64(t_codes),
                    "scale" : t_scale,
                    "zero"  : t_zero,
                },
                "q_values": {
                    "codes" : pack_b64(v_codes),
                    "scale" : v_scale,
                    "zero"  : v_zero,
                },
            }
            compressed_tracks.append(com_track)

        compressed_clip = deepcopy(clip)
        compressed_clip["animationClip"]["tracks"] = compressed_tracks

        return compressed_clip


    def decompress(self, compressed_clip) -> list:
        tracks = compressed_clip["animationClip"]["tracks"]
        decompressed_tracks = []

        for track in tracks:
            # input digestion
            times  = track["q_times"]
            values = track["q_values"]
            original_type = track["type"].replace("_quantize_b64", "")

            # Base64 unpacking
            times_codes  = unpack_b64(times["codes_b64"], self.base_type)
            values_codes = unpack_b64(values["codes_b64"], self.base_type)
            if original_type == "quaternion":
                values_codes = values_codes.reshape(-1, 4)

            # dequantizations
            dq_times  = dequantize(times_codes, times["scale"], times["zero"])
            dq_values = dequantize(values_codes, values["scale"], values["zero"])
            
            # re-normalization
            if original_type == "quaternion":
                norms = np.linalg.norm(dq_values, axis=1, keepdims=True)
                dq_values = dq_values / norms

            # pack track
            decom_track = {
                "name"   : track["name"],
                "type"   : original_type,
                "times"  : dq_times.tolist(),
                "values" : dq_values.reshape(-1).tolist(),
            }
            decompressed_tracks.append(decom_track)

        decompressed_clip = deepcopy(compressed_clip)
        decompressed_clip["animationClip"]["tracks"] = decompressed_tracks

        return decompressed_clip

