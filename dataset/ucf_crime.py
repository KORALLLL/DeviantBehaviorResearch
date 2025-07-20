from .base import VideoDS
import pathlib

class UCF_Crime(VideoDS):
    def __init__(self, root: pathlib.Path, metapath):
        super().__init__(root, metapath)
