from abc import ABC, abstractmethod


class BaseTracksCompressor(ABC):

    @abstractmethod
    def prepare(self, clip):
        """prepare the compressor (in case data outside the tracks, in clip level is required)"""
        pass

    @abstractmethod
    def compress(self, tracks):
        """tracks compression"""
        pass

    @abstractmethod
    def decompress(self, tracks):
        """tracks decompression"""
        pass