from abc import ABC, abstractmethod


class TrackCompressor(ABC):
    """Interfacce of tracks compressors"""

    @abstractmethod
    def compress(self, tracks, **kwargs) -> list:
        """
        Compressed tracks

        Parameter
        ---
        tracks: list
            A list of tracks to compress, some compressors may compress each track independently or dependently.
        
        Returms
        ---
        compressed_tracks: list
            A list of compressed tracks.
        """
        pass

    @abstractmethod
    def decompress(self, compressed_tracks, **kwargs) -> list:
        """
        Compressed tracks

        Parameter
        ---
        compressed_tracks: list
            A list of compressed tracks to decompress back to animate-able format.

        Returns
        ---
        decompressed_tracks: list
            A list of decompressed animate-able tracks.
        """
        pass
    