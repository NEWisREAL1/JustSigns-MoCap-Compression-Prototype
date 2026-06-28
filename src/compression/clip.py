from copy import deepcopy

import numpy as np

from src.compression.base import BaseTracksCompressor
from src.compression.baseline import RawBase64Compressor
from src.data import extract_tracks


class ClipCompressor:
    """Orchestrator for end-to-end mocap clip compression"""

    def __init__(
        self, 
        blendshapes_compressor : BaseTracksCompressor = None, 
        quaternions_compressor : BaseTracksCompressor = None,
        ):
        self.blendshapes_compressor = blendshapes_compressor
        self.quaternions_compressor = quaternions_compressor

    
    def compress(self, clip):
        self.blendshapes_compressor.prepare(clip)
        self.quaternions_compressor.prepare(clip)
        
        all_tracks = clip["animationClip"]["tracks"]
        blendshapes_tracks = extract_tracks(all_tracks, type=self.blendshapes_compressor.type_name)
        quaternions_tracks = extract_tracks(all_tracks, type=self.quaternions_compressor.type_name)
        
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


    def decompress(self, compressed_clip):
        blendshape_tracks_data = compressed_clip["animationClip"]["blendshape_tracks_data"]
        quaternion_tracks_data = compressed_clip["animationClip"]["quaternion_tracks_data"]

        decom_blendshape_tracks = self.blendshapes_compressor.decompress(blendshape_tracks_data)
        decom_quaternion_tracks = self.quaternions_compressor.decompress(quaternion_tracks_data)

        result = deepcopy(compressed_clip)
        result["animationClip"].pop("blendshape_tracks_data")
        result["animationClip"].pop("quaternion_tracks_data")
        result["animationClip"]["tracks"] = decom_blendshape_tracks + decom_quaternion_tracks

        return result