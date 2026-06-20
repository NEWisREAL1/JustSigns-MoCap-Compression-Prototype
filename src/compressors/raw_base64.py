import numpy as np

from src.compressors.base import TrackCompressor
from src.data import pack_b64, unpack_b64


class RawBase64Compressor(TrackCompressor):
    
    def compress(self, tracks, **kwargs) -> list:
        compressed_tracks = []

        for track in tracks:
            com_track = {
                "name"       : track["name"],
                "type"       : track["type"] + "_raw_b64",
                "times_b64"  : pack_b64(np.asarray(track["times"], dtype=np.float64)),
                "values_b64" : pack_b64(np.asarray(track["values"], dtype=np.float64)),
            }
            compressed_tracks.append(com_track)

        return compressed_tracks, None


    def decompress(self, compressed_tracks, **kwargs) -> list:
        decompressed_tracks = []

        for track in compressed_tracks:
            decom_track = {
                "name"   : track["name"],
                "type"   : track["type"].replace("_raw_b64", ""),
                "times"  : unpack_b64(track["times_b64"], np.float64).tolist(),
                "values" : unpack_b64(track["values_b64"], np.float64).tolist(),
            }
            decompressed_tracks.append(decom_track)

        return decompressed_tracks
