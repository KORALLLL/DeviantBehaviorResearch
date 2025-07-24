import torch
import pathlib
import json

class VideoDS(torch.utils.data.Dataset):
    def __init__(self, root: pathlib.Path, metapath):
        with open(metapath) as f:
            self.video_paths = [root / line.strip() for line in f]

    def __len__(self):  return len(self.video_paths)

    def __getitem__(self, idx):
        path = str(self.video_paths[idx])
        label = 0 if "Normal" in path else 1
        return {"path": path, "label": label}


class VideoJsonDS(torch.utils.data.Dataset):
    def __init__(self, root: pathlib.Path, metapath):
        self.root = root
        with open(metapath) as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = str(self.root / (self.data[idx]['video']).strip())
        label = self.data[idx]['conversations'][1]['value']
        prompt = self.data[idx]['conversations'][0]['value']
        return {"path": path, "label": label, "prompt": prompt}



if __name__ == "__main__":
    ds = VideoJsonDS(pathlib.Path("/mnt/datasets/ucf_crime"), "train.jsonl")
    print(ds[0])