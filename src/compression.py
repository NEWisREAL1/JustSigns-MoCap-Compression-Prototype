from copy import deepcopy

from src.compressors.base import TrackCompressor


class ClipCompressor:
    """
    Orchestrator for different kind of compressors to be use in various parts of MoCap clips.

    Parameter
    ---
    number_compressor, quaternion_compressor: TrackCompressor
        Compressors that will be use to compress number/blendshape tracks and quaternion tracks respectively. 
    """
    
    def __init__(self, number_compressor: TrackCompressor, quanternion_compressor: TrackCompressor):
        self.number_compressor = number_compressor
        self.quanternion_compressor = quanternion_compressor

    
    def compress(self, clip):
        tracks = clip["animationClip"]["tracks"]
        compressed_tracks = []

        for track in tracks:
            if track["type"] == "number":
                compressed_track = self.number_compressor.compress(track)
            
            elif track["type"] == "quaternion":
                compressed_track = self.quanternion_compressor.compress(track)
            
            compressed_tracks.append(compressed_track)

        compressed_clip = deepcopy(clip)
        compressed_clip["animationClip"]["tracks"] = compressed_tracks
        return compressed_clip

    
    def decompress(self, compressed_clip):
        compressed_tracks = compressed_clip["animationClip"]["tracks"]
        decompressed_tracks = []

        for track in compressed_tracks:
            if "number" in track["type"]:
                dcompressed_track = self.number_compressor.decompress(track)
            
            elif "quaternion" in track["type"]:
                dcompressed_track = self.quanternion_compressor.decompress(track)
            
            decompressed_tracks.append(dcompressed_track)

        decompressed_clip = deepcopy(compressed_clip)
        decompressed_clip["animationClip"]["tracks"] = decompressed_tracks
        return decompressed_clip