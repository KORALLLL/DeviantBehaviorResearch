import torch
import pathlib

class VideoDS(torch.utils.data.Dataset):
    def __init__(self, root: pathlib.Path, metapath):
        with open(metapath) as f:
            self.video_paths = [root / line.strip() for line in f]

    def __len__(self):  return len(self.video_paths)

    def __getitem__(self, idx):
        path = str(self.video_paths[idx])
        label = 0 if "Normal" in path else 1
        return {"path": path, "label": label}