import torch, pathlib
from tqdm import tqdm
from loguru import logger
from models import Qwen25Adapter as Model


MODEL = "Qwen2.5-VL-7B-Instruct"
MODEL_ID = f"Qwen/{MODEL}"
CACHE_DIR = ".cache/"
DATASET_PATH = "/mnt/datasets/ucf_crime"
META_PATH = "/home/kirill/DeviantBehaviorResearch/Anomaly_Test.txt"
OUTPUT_RESULTS = f"{MODEL}_basic_prompt.txt"
FPS = 2.0
MAX_NEW_TOKENS = 32
PROMPT = ("Return `1` if the video shows any deviant, abnormal or criminal "
          "behaviour; return `0` if it does not. Respond with only that single "
          "digit and nothing else.")

CONTINUE_FROM=5


class VideoDS(torch.utils.data.Dataset):
    def __init__(self, root: pathlib.Path, metapath):
        with open(metapath) as f:
            self.video_paths = [root / line.strip() for line in f]

    def __len__(self):  return len(self.video_paths)

    def __getitem__(self, idx):
        path = str(self.video_paths[idx])
        label = 0 if "Normal" in path else 1
        return {"path": path, "label": label}



if __name__ == "__main__":
    backend = Model(model_id=MODEL_ID, cache_dir=CACHE_DIR)
    ds = VideoDS(pathlib.Path(DATASET_PATH), META_PATH)
    logger.success("dataset initialised")

    if not CONTINUE_FROM:
        with open(OUTPUT_RESULTS, "w") as f_out:
            f_out.write(f"PROMPT:\n\n\n\n{PROMPT}\n\n\n\n")

    with torch.inference_mode():
        for idx in tqdm(range(len(ds)), total=len(ds)):
            try:
                sample = ds[idx]
                if CONTINUE_FROM:
                    if idx<CONTINUE_FROM: continue
                inputs = backend.encode_query(sample["path"], PROMPT, fps=FPS)
                reply  = backend.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)

                line = f"{sample['path']}\t{reply}\n"
                with open(OUTPUT_RESULTS, "a") as f_out:
                    f_out.write(line)
                logger.info(line.strip())
            except:
                logger.error(sample['path'])
