import base64
from copy import deepcopy

import numpy as np


class Base64RawCompressor:
    """
    Packs raw float arrays directly into Base64 strings without quantization.
    Provides a scientific baseline to measure pure system (formatting) compression
    by eliminating the ASCII JSON text penalty.
    """
    def __init__(self):
        pass

    def compress(self, clip):
        tracks = clip["animationClip"]["tracks"]
        compressed_tracks = []

        for track in tracks:
            times = np.asarray(track["times"], dtype=np.float32)
            if track["type"] == "quaternion":
                values = np.asarray(track["values"], dtype=np.float32).reshape((-1, 4))
            else:
                values = np.asarray(track["values"], dtype=np.float32)

            com_track = {
                "name": track["name"],
                "type": track["type"] + "_raw_b64", # Explicit routing tag for Three.js
                "times": self._pack_b64(times),
                "values": self._pack_b64(values),
            }
            compressed_tracks.append(com_track)

        compressed_clip = deepcopy(clip)
        compressed_clip["animationClip"]["tracks"] = compressed_tracks
        
        return compressed_clip

    def decompress(self, compressed_clip):
        compressed_tracks = compressed_clip["animationClip"]["tracks"]
        decompressed_tracks = []

        for track in compressed_tracks:
            # Strip the routing tag
            original_type = track["type"].replace("_raw_b64", "")
            
            times = self._unpack_b64(track["times"], np.float32)
            values = self._unpack_b64(track["values"], np.float32)
            
            decom_track = {
                "name"  : track["name"],
                "type"  : original_type,
                "times" : times.tolist(),
                "values": values.reshape(-1).tolist(),
            }
            decompressed_tracks.append(decom_track)

        decompressed_clip = deepcopy(compressed_clip)
        decompressed_clip["animationClip"]["tracks"] = decompressed_tracks
        return decompressed_clip

    # ----- Helper API ----- #
    
    def _pack_b64(self, arr):
        return base64.b64encode(arr.tobytes()).decode('ascii')

    def _unpack_b64(self, b64_str, dtype):
        return np.frombuffer(base64.b64decode(b64_str), dtype=dtype)