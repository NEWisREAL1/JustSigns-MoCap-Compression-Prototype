import numpy as np
import pandas as pd
from scipy.linalg import eigh

from src.cagd.splines import BSpline, StationarySpline


class TrackCategorizer:
    """
    Catagorize quaternions data into static and dynamic
    """    
    def __init__(self, threshold_deg=2.8, dist_agg_func=np.max):
        self.threshold_deg = threshold_deg
        self.dist_agg_func = dist_agg_func

    def categorize(self, data):
        q_mean = self.quaternions_mean(data)
        distances_deg = self.geodesic_distances_deg(data, q_mean)
        max_distance_deg = self.dist_agg_func(distances_deg)
        return max_distance_deg <= self.threshold_deg, q_mean, max_distance_deg

    def quaternions_mean(self, quats):
        accum_mat = (quats.T @ quats) / quats.shape[0]
        eigenvals, eigenvecs = eigh(accum_mat)
        return eigenvecs[:, np.argmax(eigenvals)]

    def geodesic_distances_deg(self, quats1, quats2):
        quats1 = np.atleast_2d(quats1)
        quats2 = np.atleast_2d(quats2)
        distances_rad = 2 * np.arccos(np.clip(np.abs(quats1 @ quats2.T), 0.0, 1.0))
        return np.rad2deg(distances_rad)



class MoCapCompressor:
    """
    Warpper for MoCap compression/decompression
    """

    def __init__(
        self, 
        track_cate=TrackCategorizer(2.8), 
        compress_preprocessors : list = None,
        decompress_postprocessors : list = None,
        ):
        self.compress_preprocessors = compress_preprocessors
        self.decompress_postprocessors = decompress_postprocessors
        self.track_cate = track_cate

    def compress(self, clip_quats, solver, **fit_params):
        compressed_clip_quats = {}
        info = dict(error={}, num_control_pts={})

        for joint, quats in clip_quats.items():
            # PREPROCESSING
            if self.compress_preprocessors is not None:
                for prep in self.compress_preprocessors:
                    quats = prep(quats)

            # STATIC JOINT DETECTION
            is_static, q_mean, deviation_deg = self.track_cate.categorize(quats)

            # COMPRESSING
            if is_static:
                # stationary joint -> just collpase to singular point
                compressed_clip_quats[joint] = {}
                compressed_clip_quats[joint]["compression_type"] = "singular"
                compressed_clip_quats[joint]["fitted_spl"] = StationarySpline(q_mean)
                compressed_clip_quats[joint]["data_time_size"] = quats.shape[0]
                
                info["error"][joint] = [deviation_deg]
                info["num_control_pts"][joint] = [1]

            else:
                # mobile joint -> bspline fitting
                solver.clear()
                fitted_spl, data_time = solver.fit(quats, **fit_params)

                compressed_clip_quats[joint] = {}
                compressed_clip_quats[joint]["compression_type"] = "bspline"
                compressed_clip_quats[joint]["fitted_spl"] = fitted_spl
                compressed_clip_quats[joint]["data_time"] = data_time

                info["error"][joint] = solver.history["error"]
                info["num_control_pts"][joint] = solver.history["num_control_pts"]

        return compressed_clip_quats, info

    def decompress(self, compressed_clip_quats):
        decompressed_clip_quats = {}

        for joint, data in compressed_clip_quats.items():
            type = data["compression_type"]

            if type == "singular":
                data_time = np.zeros(shape=(data["data_time_size"]))

            elif type == "bspline":
                data_time = data["data_time"]

            else:
                raise ValueError(f"Unknown compression type {type}")

            quats = data["fitted_spl"](data_time)

            # POSTPROCESSING
            if self.decompress_postprocessors is not None:
                for prep in self.decompress_postprocessors:
                    quats = prep(quats)
            
            decompressed_clip_quats[joint] = quats
        
        return decompressed_clip_quats
