import numpy as np

from src.cagd.lspia import LSPIASolver
from src.cagd.splines import BSpline
from src.compressors.base import TrackCompressor
from src.compressors.quantization import QuantizeCompressor
from src.data import pack_b64, unpack_b64


class FixedSizeBSplineCompressor(TrackCompressor):
    """
    This compressor apply B-Spline LSPIA fitting on values and perform 8-bit quantize on times.
    The B-Spline data are 16-bit quantized. 

    Critical: Assume same number of frame and same keyframe timing across all joints.
    """
    
    def __init__(self, solver):
        self.solver = solver
        self.quantizer8  = QuantizeCompressor(base_type=np.uint8)
        self.quantizer16 = QuantizeCompressor(base_type=np.uint16)


    def compress(self, tracks, **kwargs) -> list:
        compressed_tracks = []

        for track in tracks:
            # input digestion
            values = np.asarray(track["values"], dtype=np.float64)
            if track["type"] == "quaternion":
                values = values.reshape(-1, 4)
            elif track["type"] == "number":
                values = values.reshape(-1, 1)

            # values LSPIA
            spline, data_times = self.solver.fit(values)

            # quantizing bspline data
            q_cps, cps_scale, cps_zero = self.quantizer16._quantize(spline.control_pts)
            # q_knot, knot_scale, knot_zero = self.quantizer16._quantize(spline.knot_vector)

            # pack track
            com_track = {
                "name": track["name"],
                "type": track["type"] + "_bspline_b64",
                # "times": "global_times",
                "values": {
                    # "degree": spline.degree,
                    "control_pts" : {
                        "codes_b64": pack_b64(q_cps),
                        "scale": cps_scale,
                        "zero": cps_zero,
                    },
                    # "knot_vector" : {
                    #    "codes_b64": pack_b64(q_knot),
                    #    "scale": knot_scale,
                    #    "zero": knot_zero,
                    #},
                    # "data_count": data_times.shape[0],
                },
            }

            compressed_tracks.append(com_track)

        # isolating & quantizing times
        times = np.asarray(tracks[0]["times"] , dtype=np.float64)
        q_times, t_scale, t_zero = self.quantizer8._quantize(times)

        # isolating frame count (for time-param and knot regeneration)
        frame_count = data_times.shape[0]
        
        global_attrs = dict(
            degree=self.solver.degree,
            num_cps=self.solver.initial_num_cps,
            frame_count=frame_count,
            global_times={
                "codes_b64": pack_b64(q_times),
                "scale": t_scale,
                "zero": t_zero,
            },
        )

        return compressed_tracks, global_attrs


    def decompress(self, compressed_tracks, **kwargs) -> list:
        decompressed_tracks = []

        # processing global vars
        degree = kwargs["degree"]
        num_cps = kwargs["num_cps"]
        frame_count = kwargs["frame_count"]
        global_times = kwargs["global_times"]

        # generate data times and knots
        data_times = np.linspace(0, 1, frame_count)
        knot_vector = LSPIASolver(degree=degree, initial_num_cps=num_cps)._init_knot_vector(data_times)
        
        # Base64 unpacking + dequantize global times
        t_codes = unpack_b64(global_times["codes_b64"], np.uint8) 
        dp_tiems = self.quantizer8._dequantize(t_codes, global_times["scale"], global_times["zero"])

        for track in compressed_tracks:
            # input digestion
            values = track["values"]
            original_type = track["type"].replace("_bspline_b64", "")

            # Base64 unpacking
            cps_codes  = unpack_b64(values["control_pts"]["codes_b64"], np.uint16)
            # knot_codes = unpack_b64(values["knot_vector"]["codes_b64"], np.uint16)

            if original_type == "quaternion":
                cps_codes = cps_codes.reshape(-1, 4)
            elif original_type == "number":
                cps_codes = cps_codes.reshape(-1, 1)

            # dequantize spline data
            dp_cps  = self.quantizer16._dequantize(cps_codes , values["control_pts"]["scale"], values["control_pts"]["zero"])
            # dp_knot = self.quantizer16._dequantize(knot_codes, values["knot_vector"]["scale"], values["knot_vector"]["zero"])

            # reconstruct values from spline
            # knot_vector = BSpline.generate_uniform_open_knots(values["degree"], control_pts.shape[0])
            # data_times = np.linspace(0, 1, values["data_count"])
            recon_values = BSpline(degree, dp_cps, knot_vector)(data_times)
            
            # re-normalization
            if original_type == "quaternion":
                norms = np.linalg.norm(recon_values, axis=1, keepdims=True)
                recon_values = recon_values / norms

            # pack track
            decom_track = {
                "name"   : track["name"],
                "type"   : original_type,
                "times"  : dp_tiems.tolist(),
                "values" : recon_values.reshape(-1).tolist(),
            }

            decompressed_tracks.append(decom_track)

        return decompressed_tracks
