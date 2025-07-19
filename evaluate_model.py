import torch, pathlib, yaml, argparse
from tqdm import tqdm
from loguru import logger
import time
import numpy as np
from scipy.stats import t

def mean_ci_halfwidth(values, alpha=0.05):

    x = np.asarray(values, dtype=float)
    mean = x.mean()
    se   = x.std(ddof=1) / np.sqrt(len(x))        
    h    = t.ppf(1 - alpha/2, df=len(x) - 1) * se 
    return mean, h

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
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()


    with open(args.config, 'r') as f: cfg = yaml.safe_load(f)

    MODEL = cfg['model']
    MODEL_ID = f"{cfg['model_space']}/{MODEL}"
    CACHE_DIR = cfg['cache_dir']
    DATASET_PATH = cfg['dataset_path']
    META_PATH = cfg['meta_path']
    OUTPUT_RESULTS = f"{MODEL}_{cfg['output_prefix']}.txt"
    FPS = cfg['fps']
    NUM_FRAMES = cfg['num_frames']
    MAX_NEW_TOKENS = cfg['max_new_tokens']

    with open(cfg['prompt'], 'r') as file: PROMPT = file.read().strip()
    logger.info(str(cfg))
    logger.info(f"Prompt: {PROMPT}")

    CONTINUE_FROM=cfg['continue_from']

    if cfg['model_space']=="LanguageBind":
        from models import VideoLlavaAdapter as Model
        logger.info("llava imported")
    elif cfg['model_space']=="google":
        from models import GemmaAdapter as Model
        logger.info("gemma imported")
    elif cfg['model_space']=="Qwen":
        from models import Qwen25Adapter as Model
        logger.info("qwen imported")
    elif cfg["model_space"]=="OpenGVLab":
        from models import InternVL3Adapter as Model
        logger.info("intervl imported")
    else: raise NotImplementedError


    backend = Model(model_id=MODEL_ID, cache_dir=CACHE_DIR)
    ds = VideoDS(pathlib.Path(DATASET_PATH), META_PATH)
    logger.success("dataset initialised")

    if not CONTINUE_FROM:
        with open(OUTPUT_RESULTS, "w") as f_out:
            f_out.write(f"PROMPT:\n\n\n\n{PROMPT}\n\n\n\n")
    results = []
    with torch.inference_mode():
        for idx in tqdm(range(len(ds)), total=len(ds)):
            try:
                start = time.time()
                sample = ds[idx]
                if CONTINUE_FROM:
                    if idx<CONTINUE_FROM: continue
                inputs = backend.encode_query(sample["path"], PROMPT, fps=FPS, num_frames=NUM_FRAMES)
                reply  = backend.generate(inputs, max_new_tokens=MAX_NEW_TOKENS)
                end = time.time()

                line = f"{sample['path']}\t{reply}\n"
                with open(OUTPUT_RESULTS, "a") as f_out:
                    f_out.write(line)
                logger.info(line.strip())
                logger.info(str(end-start))
                results.append(end-start)

            except:
                logger.error(sample['path'])


    m, hw = mean_ci_halfwidth(results)
    logger.info(f"{m:.4f} ±{hw:.4f}")
