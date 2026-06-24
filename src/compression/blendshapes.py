import numpy as np

from src.compression.utils import dequantize, pack_b64, quantize, unpack_b64


class BlendShapesSchemeCompressor:

    def __init__(
        self, 
        type_name="number", 
        quantize_type=np.uint8, 
        frame_code_type=np.uint16, 
        frame_code_fps=120,
        decimation_thres=0,
        ):
        self.type_name = type_name
        self.q_type = quantize_type
        self.f_type = frame_code_type
        self.fps = frame_code_fps
        self.decimation_thres = decimation_thres


    def compress(self, tracks):
        tracks_data = dict(
            compress_type = "blendshapes_scheme",
            type_name = self.type_name,
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

            # colinear decimation (on reduced data)
            dec_times_f_codes, dec_val_codes = self._colinear_decimation(times_f_codes, val_codes)
            
            # pack group data
            tracks_data["groups"].append(dict(
                names = group["names"],
                f_times_b64 = pack_b64(dec_times_f_codes.astype(self.f_type)),
                q_values = dict(
                    codes_b64 = pack_b64(dec_val_codes.astype(self.q_type)),
                    scale     = val_scale,
                    zero      = val_zero,
                ),
            ))

        return tracks_data


    def decompress(self, tracks_data):
        decompressed_tracks = []

        type_name = tracks_data["type_name"]
        groups = tracks_data["groups"]
        f_fps = tracks_data["f_fps"]

        for group in groups:
            # base64 unpackings
            f_times = unpack_b64(group["f_times_b64"], self.f_type)
            q_values = group["q_values"]
            v_codes = unpack_b64(q_values["codes_b64"], self.q_type)

            # decoding times indices
            times = self._frame_deindexing(f_times, f_fps)

            # dequantizing values
            values = dequantize(v_codes, q_values["scale"], q_values["zero"])

            for name in group["names"]:
                decom_track = {
                    "name"   : name,
                    "type"   : type_name,
                    "times"  : times.tolist(),
                    "values" : values.tolist(),
                }
                decompressed_tracks.append(decom_track)

        return decompressed_tracks


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

    def _frame_deindexing(self, f_codes, fps=None):
        if fps is None:
            fps = self.f_type
        return f_codes / fps


    # COLINEAR DECIMATION

    def _colinear_decimation(self, times, values):
        vecs = np.stack([times, values], axis=1)
        i = 1
        while i < vecs.shape[0] - 1:
                v = vecs[i + 1] - vecs[i - 1]
                w = vecs[i] - vecs[i - 1]
                projection_weight = np.dot(w, v) / np.linalg.norm(v) ** 2
                perpendicular_vec = w - projection_weight * v
                perpendicular_dist = np.linalg.norm(perpendicular_vec)

                if perpendicular_dist <= self.decimation_thres:
                    vecs = np.delete(vecs, i, axis=0)
                else:
                    i += 1
                
        return vecs[:, 0], vecs[:, 1]