import cv2
from PIL import Image
from .base import VLMBackend

from loguru import logger
import torch
import tempfile
from pathlib import Path

def extract_frames(video_path: str, num_frames: int):
    """
    The function is adapted from:
    https://github.com/merveenoyan/smol-vision/blob/main/Gemma_3_for_Video_Understanding.ipynb
    """
    video_path = video_path if video_path.endswith("mp4") else f"{video_path}.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        assert False
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the step size to evenly distribute frames across the video.
    step = total_frames // num_frames
    frames = []

    for i in range(num_frames):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp = round(frame_idx / fps, 2)
        frames.append((img, timestamp))

    cap.release()
    return frames

class Qwen25Adapter(VLMBackend):
    def __init__(self, model_id: str, cache_dir: str, **kwargs):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).eval()
        logger.success("model loaded")

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.process_vision_info = process_vision_info
        logger.success("processor inititalised")

    def encode_query(self, video_path: str, prompt: str, fps:float=1.0, num_frames:int=16, **kwargs):
        video_path = video_path if video_path.endswith(".mp4") else f"{video_path}.mp4"
        video_frames = extract_frames(video_path, num_frames=num_frames)
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": [], "fps": fps},
                {"type": "text",  "text": prompt.strip()}
            ]
        }]

        # video as a sequence of images 
        # see https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct#using-🤗--transformers-to-chat 
        # at Video inference
        temp_dir = tempfile.TemporaryDirectory()
        for frame_data in video_frames:
            img, timestamp = frame_data
            img.save(f"{Path(temp_dir.name)}/frame_{timestamp}.png")
            messages[0]["content"][0]["video"].append(f"{Path(temp_dir.name)}/frame_{timestamp}.png")

        chat = self.processor.apply_chat_template(messages,
                                                  tokenize=False,
                                                #   num_frames=num_frames,
                                                  add_generation_prompt=True)

        img_inp, vid_inp, vid_kwargs = self.process_vision_info(messages,
                                                                return_video_kwargs=True)
        pr = self.processor(
            text=[chat],
            images=img_inp,
            videos=vid_inp,
            padding=True,
            return_tensors="pt",
            **vid_kwargs,
        )
        return pr.to(self.model.device)

    def generate(self, inputs, max_new_tokens: int) -> str:
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(
            out[:, inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )[0]
