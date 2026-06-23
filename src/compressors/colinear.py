import numpy as np

from src.compressors.base import TrackCompressor
from src.compressors.quantization import QuantizeCompressor
from src.data import pack_b64, unpack_b64


class ColinearDecimatoinCompressor(TrackCompressor):
    """
    Perform Colinear Decimation and quantization for number tracks.

    This compression shared the same decoder as QuantizationCompressor.
    """
    
    def __init__(self):
        self.quantizer8  = QuantizeCompressor(base_type=np.uint8)


    def compress(self, tracks, **kwargs) -> list:
        compressed_tracks = []

        for track in tracks:
            if track["type"] == "quaternion":
                raise ValueError("Colinear Decimation only compatible for \"number\" type.")
            
            # input digestion
            times  = np.asarray(track["times"], dtype=np.float64)
            values = np.asarray(track["values"], dtype=np.float64)

            i = 1
            while i < len(values) - 1:
                perpen_dist = self._point_line_distance(
                    [times[i - 1], values[i - 1]],
                    [times[i + 0], values[i + 0]],
                    [times[i + 1], values[i + 1]],
                )

                if perpen_dist <= 1e-3:
                    # colinear -> remove
                    times = np.delete(times, i)
                    values = np.delete(values, i)
                else:
                    # not-colinear -> continue
                    i += 1

            # quantization
            t_codes, t_scale, t_zero = self.quantizer8._quantize(times)
            v_codes, v_scale, v_zero = self.quantizer8._quantize(values)

            # pack track
            com_track = {
                "name": track["name"],
                "type": track["type"] + "_quantize_b64",
                "times": {
                    "codes_b64": pack_b64(t_codes),
                    "scale": t_scale,
                    "zero": t_zero,
                },
                "values": {
                    "codes_b64": pack_b64(v_codes),
                    "scale": v_scale,
                    "zero": v_zero,
                },
            }

            compressed_tracks.append(com_track)

        return compressed_tracks, None


    def decompress(self, compressed_tracks, **kwargs) -> list:
        return self.quantizer8.decompress(compressed_tracks, **kwargs)


    # ----- Helper API ----- #

    def _point_line_distance(self, q, p1, p2):
        q, p1, p2 = np.array(q), np.array(p1), np.array(p2)
        v = p2 - p1
        w = q - p1
        projection_factor =  np.dot(w, v) / (np.linalg.norm(v) ** 2)
        w_perpendicular = w - projection_factor * v
        return np.linalg.norm(w_perpendicular) 