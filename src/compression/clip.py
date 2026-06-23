from copy import deepcopy

import numpy as np

from src.compression.baseline import RawBase64Compressor
from src.data import extract_tracks


class ClipCompressor:
    """Orchestrator for end-to-end mocap clip compression"""

    def __init__(self, blendshapes_compressor=None, quaternions_compressor=None):
        self.blendshapes_compressor = blendshapes_compressor
        self.quaternions_compressor = quaternions_compressor

    
    def compress(self, clip):
        all_tracks = clip["animationClip"]["tracks"]
        blendshapes_tracks = extract_tracks(all_tracks, type="number")
        quaternions_tracks = extract_tracks(all_tracks, type="quaternion")
        
        if self.blendshapes_compressor is not None:
            blendshape_tracks_data = self.blendshapes_compressor.compress(blendshapes_tracks)
        else:
            blendshape_tracks_data = RawBase64Compressor().compress(blendshapes_tracks, pass_tracks=True)
        
        if self.quaternions_compressor is not None:
            quaternion_tracks_data = self.quaternions_compressor.compress(quaternions_tracks)
        else:
            quaternion_tracks_data = RawBase64Compressor().compress(quaternions_tracks, pass_tracks=True)

        result = deepcopy(clip)
        result["animationClip"].pop("tracks", None)
        result["animationClip"]["blendshape_tracks_data"] = blendshape_tracks_data
        result["animationClip"]["quaternion_tracks_data"] = quaternion_tracks_data

        return result


    def decompress(self, clip):
        pass