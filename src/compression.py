from copy import deepcopy

from src.compressors.base import TrackCompressor
from src.data import extract_tracks


class ClipCompressor:
    """
    Orchestrator for different kind of tracks compressors to be use in various parts of MoCap clips.

    Parameter
    ---
    number_compressor, quaternion_compressor: TrackCompressor
        Compressors that will be use to compress number/blendshape tracks and quaternion tracks respectively. 
    """
    
    def __init__(
        self, 
        number_compressor: TrackCompressor, 
        quanternion_compressor: TrackCompressor,
        preprocessors: list = None,
        ):
        self.number_compressor = number_compressor
        self.quanternion_compressor = quanternion_compressor
        self.preprocessors = preprocessors

    def compress(self, clip):
        if self.preprocessors is not None:
            clip = self.preprocess(clip)

        tracks = clip["animationClip"]["tracks"]
        number_tracks = extract_tracks(tracks, type="number")
        quaternion_tracks = extract_tracks(tracks, type="quaternion")
        
        compressed_num_tracks, num_global_attrs = self.number_compressor.compress(number_tracks)
        compressed_qtn_tracks, qtn_global_attrs = self.quanternion_compressor.compress(quaternion_tracks)

        combined_tracks = compressed_num_tracks + compressed_qtn_tracks
        compressed_clip = deepcopy(clip)
        compressed_clip["animationClip"]["tracks"] = combined_tracks

        if num_global_attrs is not None:
            compressed_clip["animationClip"]["number_global_attrs"] = {}
            for key, attr in num_global_attrs.items():
                compressed_clip["animationClip"]["number_global_attrs"][key] = attr

        if qtn_global_attrs is not None:
            compressed_clip["animationClip"]["quaternion_global_attrs"] = {}
            for key, attr in qtn_global_attrs.items():
                compressed_clip["animationClip"]["quaternion_global_attrs"][key] = attr

        return compressed_clip

    
    def decompress(self, compressed_clip):
        tracks = compressed_clip["animationClip"]["tracks"]
        number_tracks = extract_tracks(tracks, type="number")
        quaternion_tracks = extract_tracks(tracks, type="quaternion")
        
        if "number_global_attrs" in compressed_clip["animationClip"].keys():
            decompressed_num_tracks = self.number_compressor.decompress(
                number_tracks, 
                **compressed_clip["animationClip"]["number_global_attrs"]
                )
        else:
            decompressed_num_tracks = self.number_compressor.decompress(number_tracks)
        
        if "quaternion_global_attrs" in compressed_clip["animationClip"].keys():
            decompressed_qtn_tracks = self.quanternion_compressor.decompress(
                quaternion_tracks, 
                **compressed_clip["animationClip"]["quaternion_global_attrs"]
                )
        else:
            decompressed_qtn_tracks = self.quanternion_compressor.decompress(quaternion_tracks)
        
        combined_tracks = decompressed_num_tracks + decompressed_qtn_tracks
        decompressed_clip = deepcopy(compressed_clip)
        decompressed_clip["animationClip"]["tracks"] = combined_tracks
        return decompressed_clip


    def preprocess(self, clip):
        tracks = clip["animationClip"]["tracks"]
        preped_tracks = []

        for track in tracks:
            preped_t = None
            for prep in self.preprocessors:
                preped_t = prep.process(track)
            preped_tracks.append(preped_t)

        preped_clip = deepcopy(clip)
        preped_clip["animationClip"]["tracks"] = preped_tracks
        return preped_clip
        