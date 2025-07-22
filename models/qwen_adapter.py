from .base import VLMBackend

from loguru import logger
from peft import PeftModel

class Qwen25AdapterLoRA(VLMBackend):
    def __init__(self, model_id: str, cache_dir: str, **kwargs):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
        ).eval()
        logger.success("model loaded")
        self.model = PeftModel.from_pretrained(self.model, kwargs["adapter_id"])

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.process_vision_info = process_vision_info
        logger.success("processor inititalised")

    def encode_query(self, video_path: str, prompt: str, fps=1.0, **kwargs):
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "text",  "text": prompt.strip()}
            ]
        }]
        chat = self.processor.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)

        img_inp, vid_inp, vid_kwargs = self.process_vision_info(messages,
                                                                return_video_kwargs=True)

        return self.processor(
            text=[chat],
            images=img_inp,
            videos=vid_inp,
            padding=True,
            return_tensors="pt",
            **vid_kwargs,
        ).to(self.model.device)

    def generate(self, inputs, max_new_tokens: int) -> str:
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.processor.batch_decode(
            out[:, inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )[0]
