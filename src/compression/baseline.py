from copy import deepcopy

import numpy as np

from src.compression.utils import dequantize, pack_b64, quantize, unpack_b64


class RawBase64Compressor:
    def __init__(self, type_name, type_validate=True):
        self.compress_type = "raw_base64"
        self.type_name = type_name
        self.type_validate = type_validate
    
    
    def compress(self, tracks) -> list:
        if self.type_validate:
            for track in tracks:
                if track["type"] != self.type_name:
                    raise ValueError(f"Mismatch type name found, expected {self.type_name}, got {track["type"]}.")
        
        tracks_data = dict(
            compression_type = self.compress_type,
            type_name = self.type_name,
            tracks = [],
        )

        for track in tracks:
            com_track = {
                "name"       : track["name"],
                "times_b64"  : pack_b64(np.asarray(track["times"], dtype=np.float64)),
                "values_b64" : pack_b64(np.asarray(track["values"], dtype=np.float64)),
            }
            tracks_data["tracks"].append(com_track)
    
        return tracks_data


    def decompress(self, tracks_data) -> list:
        decompressed_tracks = []

        type_name = tracks_data["type_name"]
        tracks = tracks_data["tracks"]

        for track in tracks:
            decom_track = {
                "name"   : track["name"],
                "type"   : type_name,
                "times"  : unpack_b64(track["times_b64"], np.float64).tolist(),
                "values" : unpack_b64(track["values_b64"], np.float64).tolist(),
            }
            decompressed_tracks.append(decom_track)

        return decompressed_tracks


class QuantizeCompressor:

    def __init__(self, type_name, q_type=np.uint8, type_validate=True):
        self.compress_type = "direct_quantize"
        self.type_name = type_name
        self.q_type = q_type
        self.type_validate = type_validate
    
    
    def compress(self, tracks) -> list:
        if self.type_validate:
            for track in tracks:
                if track["type"] != self.type_name:
                    raise ValueError(f"Mismatch type name found, expected {self.type_name}, got {track["type"]}.")
        
        tracks_data = dict(
            compression_type = self.compress_type,
            type_name = self.type_name,
            q_bits = np.dtype(self.q_type).itemsize * 8,
            tracks = [],
        )

        for track in tracks:
            t_codes, t_scale, t_zero = quantize(np.array(track["times"]), self.q_type)
            v_codes, v_scale, v_zero = quantize(np.array(track["values"]), self.q_type)

            com_track = {
                "name": track["name"],
                "q_times": {
                    "codes_b64": pack_b64(t_codes),
                    "scale": t_scale,
                    "zero": t_zero,
                },
                "q_values": {
                    "codes_b64": pack_b64(v_codes),
                    "scale": v_scale,
                    "zero": v_zero,
                },
            }
            tracks_data["tracks"].append(com_track)
    
        return tracks_data


    def decompress(self, tracks_data) -> list:
        decompressed_tracks = []

        type_name = tracks_data["type_name"]
        tracks = tracks_data["tracks"]

        for track in tracks:
            q_times = track["q_times"]
            decode_t_codes = unpack_b64(q_times["codes_b64"], self.q_type)
            dq_times = dequantize(decode_t_codes, q_times["scale"], q_times["zero"])
            
            q_values = track["q_values"]
            decode_v_codes = unpack_b64(q_values["codes_b64"], self.q_type)
            dq_values = dequantize(decode_v_codes, q_values["scale"], q_values["zero"])

            decom_track = {
                "name"   : track["name"],
                "type"   : type_name,
                "times"  : dq_times.tolist(),
                "values" : dq_values.tolist(),
            }
            decompressed_tracks.append(decom_track)

        return decompressed_tracks

