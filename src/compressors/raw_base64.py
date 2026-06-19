import numpy as np

from src.compressors.base import TrackCompressor
from src.data import pack_b64, unpack_b64


class RawBase64Compressor(TrackCompressor):
    
    def compress(self, track):
        return {
            "name"   : track["name"],
            "type"   : track["type"] + "_raw_b64",
            "times"  : pack_b64(np.asarray(track["times"] , dtype=np.float64)),
            "values" : pack_b64(np.asarray(track["values"], dtype=np.float64)),
        }

    def decompress(self, compressed_track):
        return {
            "name"   : compressed_track["name"],
            "type"   : compressed_track["type"].replace("_raw_b64", ""),
            "times"  : unpack_b64(compressed_track["times"] , np.float64).tolist(),
            "values" : unpack_b64(compressed_track["values"], np.float64).tolist(),
        }
