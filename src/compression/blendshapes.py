import numpy as np

from src.compression.utils import dequantize, pack_b64, quantize, unpack_b64


class BlendShapesCompressor:

    def __init__(self, type_name="number", quantize_type=np.uint8, frame_code_type=np.uint16, frame_code_fps=120):
        self.type_name = type_name
        self.q_type = quantize_type
        self.f_type = frame_code_type
        self.fps = frame_code_fps


    def compress(self, tracks):
        blendshape_tracks_data = dict(
            q_bits = np.dtype(self.q_type).itemsize * 8,
            f_fps  = int(self.fps),
            groups = [],
            )

        unique_group = self._get_unique_groups(tracks)

        for group in unique_group:
            times = group["times"]
            values = group["values"]

            # delete (skip) dead track
            if np.allclose(np.array(values), 0):
                continue

            # times: frame index encoding
            times_f_codes = self._frame_indexing(np.array(times))

            # values: quantization
            val_codes, val_scale, val_zero = quantize(values, q_type=self.q_type)

            # pack group data
            blendshape_tracks_data["groups"].append(dict(
                type = self.type_name,
                names = group["names"],
                f_times = pack_b64(times_f_codes),
                q_values = dict(
                    codes = pack_b64(val_codes),
                    scale = val_scale,
                    zero  = val_zero,
                ),
            ))

        return blendshape_tracks_data


    def decompress(self, compressed_tracks_data):
        pass


    # ----- HELPER API'S ----- #

    def _get_unique_groups(self, tracks):
        unique_group = []

        for track in tracks:
            name, times, values = track["name"], track["times"], track["values"]

            if not unique_group:
                unique_group.append(dict(names=[name], times=times, values=values))

            else:
                dup = False
                for data in unique_group:
                    if len(times) == len(data["times"]):
                        if np.allclose(times, data["times"]) & np.allclose(values, data["values"]):
                            data["names"].append(name)
                            dup = True
                
                if not dup:
                    unique_group.append(dict(names=[name], times=times, values=values))

        return unique_group

    # TIMES FRAME INDEX CODING
    
    def _frame_indexing(self, times):
        return np.round(times * self.fps).astype(self.f_type)

    def _frame_deindexing(self, f_codes):
        return f_codes / self.fps