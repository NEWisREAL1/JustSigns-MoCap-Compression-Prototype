from copy import deepcopy

import numpy as np
from scipy.spatial import geometric_slerp


class AntipodalPreprocessor:
    """Ensure there are no sudden sign flips in the sequence of quaternions"""

    def process(self, track):
        prep_track = deepcopy(track)

        if "quaternion" in track["type"]:
            aligned_values = np.copy(prep_track["values"]).reshape(-1, 4)
            for i in range(1, aligned_values.shape[0]):
                if np.dot(aligned_values[i], aligned_values[i - 1]) < 0:
                    aligned_values[i] = -aligned_values[i]
            prep_track["values"] = aligned_values.reshape(-1)

        return prep_track


class TrackBaker:
    """Bake the track to regular interval with given FPS"""

    def __init__(self, lerp_type, fps, duration):
        if lerp_type not in ['lerp', 'slerp']:
            raise ValueError(f"invalid lerp type `{lerp_type}`, expected `lerp` pr `slerp`")
        
        self.lerp_type = lerp_type
        self.fps = fps
        self.duration = duration
        self.times = np.linspace(0, duration, np.ceil(fps * duration).astype(int) + 1)


    def process(self, track):
        track_times = np.array(track["times"])
        track_values = np.array(track["values"])

        if track["type"] == "quaternion":
            # assume from slerp that the values are quaternions
            prep_values =np.zeros(shape=(len(self.times), 4))
            track_values = track_values.reshape(-1, 4)
        else:
            prep_values = np.zeros(shape=(len(self.times)))


        for i, t in enumerate(self.times):
            times_diff = track_times - t
            
            exact_idx = np.where(times_diff == 0)[0]
            if len(exact_idx) > 0:
                exact_idx = exact_idx[0]
                prep_values[i] = track_values[exact_idx]

            else:
                insert_idx = np.searchsorted(track_times, t)
                idx_1 = insert_idx - 1 if insert_idx > 0 else 0
                idx_2 = insert_idx if insert_idx < track_times.shape[0] - 1 else track_times.shape[0] - 1

                if idx_1 == idx_2:
                    lerp_t = 0
                else:
                    start = track_values[idx_1]
                    end = track_values[idx_2]
                    lerp_t = (t - track_times[idx_1]) / (track_times[idx_2] - track_times[idx_1])

                if self.lerp_type == 'lerp':
                    prep_values[i] = self._lerp(start, end, lerp_t)
                elif self.lerp_type == 'slerp':
                    prep_values[i] = self._slerp(start, end, lerp_t)

        return {
            "name": track["name"],
            "type": track["type"],
            "times": self.times,
            "values": prep_values.reshape(-1),
        }


    # ----- Helper API ----- #

    def _lerp(self, start, end, t):
        return (1 - t) * start + t * end

    def _slerp(self, start, end, t):
        return geometric_slerp(start / np.linalg.norm(start), end / np.linalg.norm(end), t)
