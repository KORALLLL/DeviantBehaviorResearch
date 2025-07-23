import torch
import pathlib
import pandas as pd
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
        self.df = pd.read_json(metapath, lines=True)
        self.root = root

    def __len__(self):  return len(self.df)

    def _find_in_conversation(self, row, key):
        for message in row['conversations']:
            if message['from'] == key:
                return message['value']
        return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(self.root / row['video'])
        prompt = self._find_in_conversation(row, 'human')
        label = self._find_in_conversation(row, 'gpt')
        return {"path": path, "prompt": prompt, "label": label}