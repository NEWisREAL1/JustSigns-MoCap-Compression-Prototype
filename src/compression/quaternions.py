import numpy as np
from scipy.linalg import eigh
from scipy.spatial.transform import Rotation as R

from src.cagd.lspia.solver import LSPIASolver
from src.compression.base import BaseTracksCompressor
from src.compression.utils import (
    LEFT_HAND,
    RIGHT_HAND,
    dequantize,
    pack_b64,
    quantize,
    unpack_b64,
)
from src.processing import AntipodalPreprocessor, TrackBaker


class QuaternionsSchemeCompressor(BaseTracksCompressor):

    def __init__(
        self, 
        type_name="quaternion", 
        quantize_type=np.uint8,
        bake_fps=30,
        residuals_subsampling_fps=10,
        pca_on_hands=True,
        solver : LSPIASolver = None,
        ):
        self.type_name = type_name
        self.q_type = quantize_type
        self.bake_fps = bake_fps
        self.sub_fps = residuals_subsampling_fps
        self.pca_on_hands = pca_on_hands
        self.solver = solver

        self.antipodal = AntipodalPreprocessor()

    
    def prepare(self, clip):
        self.duration = clip["animationClip"]["duration"]
        self.baker = TrackBaker(lerp_type='slerp', fps=self.bake_fps, duration=self.duration)
        self.subsampler = TrackBaker(lerp_type='lerp', fps=self.sub_fps, duration=self.duration)


    def compress(self, tracks):
        tracks_data = dict(
            compress_type = "quaternions_scheme",
            type_name = self.type_name,
            q_bits = np.dtype(self.q_type).itemsize * 8,
            bake_fps = self.bake_fps,
            res_fps = self.sub_fps,
            tracks = [],
            eigen_tracks = [],
            static_tracks = [],
            )

        left_hand_joints, right_hand_joints = [], []
        left_hand_signals, right_hand_signals = None, None
        
        for track in tracks:
            # preprocessing: antipodal alignment & baking
            aligned_track = self.antipodal.process(track)        
            baked_track = self.baker.process(aligned_track)
            
            values = np.reshape(baked_track["values"], shape=(-1, 4))
            times = np.array(baked_track["times"])

            if self.pca_on_hands:
                # save for PCA processing later
                if track["name"].replace(".quaternion", "") in LEFT_HAND:
                    left_hand_joints.append(track["name"])
                    if left_hand_signals is None:
                        left_hand_signals = values
                    else:
                        left_hand_signals = np.concatenate((left_hand_signals, values), axis=1)
                    continue
                
                elif track["name"].replace(".quaternion", "") in RIGHT_HAND:
                    right_hand_joints.append(track["name"])
                    if right_hand_signals is None:
                        right_hand_signals = values
                    else:
                        right_hand_signals = np.concatenate((right_hand_signals, values), axis=1)
                    continue
            
            # determine whether the track is static
            rotations = R.from_quat(values)
            mean_rot = rotations.mean()
            rot_deviations = mean_rot.inv() * rotations
            max_deviation_rad = np.max(rot_deviations.magnitude())
            
            if max_deviation_rad < np.deg2rad(5):
                tracks_data["static_tracks"].append(dict(
                    name = track["name"],
                    value = mean_rot.as_quat(),
                ))

            else:
                # bspline fitting
                spline, abscissas = self.solver.fit(values, times)

                # residuals calculation and compression
                residuals = values - spline(abscissas)
                subsampled_residuals = self.subsampler.process(dict(
                    name="temp",
                    type="quaternion",
                    values=residuals.reshape(-1),
                    times=times,
                ))["values"].reshape(-1, 4)

                ### quantizations
                # control points
                x_cps_codes, x_cps_scale, x_cps_zero = quantize(spline.control_points[:, 0], self.q_type)
                y_cps_codes, y_cps_scale, y_cps_zero = quantize(spline.control_points[:, 1], self.q_type)
                z_cps_codes, z_cps_scale, z_cps_zero = quantize(spline.control_points[:, 2], self.q_type)
                w_cps_codes, w_cps_scale, w_cps_zero = quantize(spline.control_points[:, 3], self.q_type)
                # knots
                knots_codes, knots_scale, knots_zero = quantize(spline.knot_vector, self.q_type)
                # residuals
                x_res_codes, x_res_scale, x_res_zero = quantize(subsampled_residuals[:, 0], self.q_type)
                y_res_codes, y_res_scale, y_res_zero = quantize(subsampled_residuals[:, 1], self.q_type)
                z_res_codes, z_res_scale, z_res_zero = quantize(subsampled_residuals[:, 2], self.q_type)
                w_res_codes, w_res_scale, w_res_zero = quantize(subsampled_residuals[:, 3], self.q_type)

                ### Baed64 encoding
                x_cps_codes_b64 = pack_b64(x_cps_codes)
                y_cps_codes_b64 = pack_b64(y_cps_codes)
                z_cps_codes_b64 = pack_b64(z_cps_codes)
                w_cps_codes_b64 = pack_b64(w_cps_codes)

                knots_codes_b64 = pack_b64(knots_codes)

                x_res_codes_b64 = pack_b64(x_res_codes)
                y_res_codes_b64 = pack_b64(y_res_codes)
                z_res_codes_b64 = pack_b64(z_res_codes)
                w_res_codes_b64 = pack_b64(w_res_codes)

                # pack track data
                tracks_data["tracks"].append(dict(
                    name = track["name"],
                    control_points = dict(
                        x = dict(codes_b64=x_cps_codes_b64, scale=x_cps_scale, zero=x_cps_zero),
                        y = dict(codes_b64=y_cps_codes_b64, scale=y_cps_scale, zero=y_cps_zero),
                        z = dict(codes_b64=z_cps_codes_b64, scale=z_cps_scale, zero=z_cps_zero),
                        w = dict(codes_b64=w_cps_codes_b64, scale=w_cps_scale, zero=w_cps_zero),
                    ),
                    konts = dict(codes_b64=knots_codes_b64, scale=knots_scale, zero=knots_zero),
                    residuals = dict(
                        x = dict(codes_b64=x_res_codes_b64, scale=x_res_scale, zero=x_res_zero),
                        y = dict(codes_b64=y_res_codes_b64, scale=y_res_scale, zero=y_res_zero),
                        z = dict(codes_b64=z_res_codes_b64, scale=z_res_scale, zero=z_res_zero),
                        w = dict(codes_b64=w_res_codes_b64, scale=w_res_scale, zero=w_res_zero),
                    ),
                ))

        ### PCA
        
        if self.pca_on_hands:
            hands_joints = [left_hand_joints, right_hand_joints]
            hands_signals = [left_hand_signals, right_hand_signals]

            for joint_names, signals in zip(hands_joints, hands_signals):
                # mean & centering
                hand_mean  = np.mean(signals, axis=0)
                signals_cen  = signals - hand_mean

                # cov, eigenvecs, eigenvals
                hand_cov  = np.cov(signals_cen, rowvar=False)
                hand_eigenvals , hand_eigenvecs  = eigh(hand_cov)

                # eigenvals sorting
                idx = np.argsort(hand_eigenvals)[::-1]
                hand_eigenvecs = hand_eigenvecs[:, idx]
                hand_eigenvals = hand_eigenvals[idx]

                # PVE, cutting, scoring
                hand_pve = np.cumsum(hand_eigenvals) / np.sum(hand_eigenvals)
                top_pcs_idx  = np.clip(np.searchsorted(hand_pve , 0.99), 1, hand_eigenvecs.shape[1])
                hand_top_pcs  = hand_eigenvecs[:, :top_pcs_idx]
                hand_pc_scores  = np.dot(signals_cen, hand_top_pcs)

                tracks_data["eigen_tracks"].append(dict(
                    names = joint_names,
                    num_pcs = top_pcs_idx,
                    eigenvecs = [],
                    pc_scores = [],
                    mean = None,
                ))

                for pc in hand_top_pcs.T:
                    codes, scale, zero = quantize(pc, self.q_type)
                    codes = pack_b64(codes)
                    tracks_data["eigen_tracks"][-1]["eigenvecs"].append(dict(
                        codes_b64 = codes,
                        scale = scale,
                        zero = zero,
                    ))

                for pc_score in hand_pc_scores.T:
                    codes, scale, zero = quantize(pc_score, self.q_type)
                    codes = pack_b64(codes)
                    tracks_data["eigen_tracks"][-1]["pc_scores"].append(dict(
                        codes_b64 = codes,
                        scale = scale,
                        zero = zero,
                    ))


                codes, scale, zero = quantize(hand_mean, self.q_type)
                codes = pack_b64(codes)
                tracks_data["eigen_tracks"][-1]["mean"] = dict(
                        codes_b64 = codes,
                        scale = scale,
                        zero = zero,
                )

        return tracks_data


    def decompress(self, tracks_data):
        return

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
