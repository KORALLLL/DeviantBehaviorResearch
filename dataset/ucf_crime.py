from .base import VideoJsonDS
import pathlib

class UCF_Crime(VideoJsonDS):
    def __init__(self, root: pathlib.Path, metapath):
        super().__init__(root, metapath)
