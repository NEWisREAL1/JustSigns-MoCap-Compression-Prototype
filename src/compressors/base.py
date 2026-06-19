from abc import ABC, abstractmethod


class TrackCompressor(ABC):

    @abstractmethod
    def compress(self, track):
        pass

    @abstractmethod
    def decompress(self, compressed_track):
        pass
    