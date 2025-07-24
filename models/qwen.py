from .base import VLMBackend

from loguru import logger
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

class Qwen25Adapter(VLMBackend):
    def __init__(self, model_id: str, cache_dir: str, quantization_config: BitsAndBytesConfig = None):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True,
            quantization_config=quantization_config,
        ).eval()
        logger.success("model loaded")

        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        self.process_vision_info = process_vision_info
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
        )
        logger.success("processor inititalised")

    @staticmethod
    def format_inference_data(video_path, prompt, fps, nframes):
        return [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "nframes": nframes},
                    {"type": "text",  "text": prompt.strip()}
                ]
        }]

    @staticmethod
    def format_finetune_data(video_path, prompt, fps, nframes, label):
        return [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "nframes": nframes},
                    {"type": "text",  "text": prompt.strip()}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text",  "text": str(label)}
                ]
            }
        ]

    def encode_query(self, video_path: str, prompt: str, fps=1.0, nframes=16, **kwargs):
        messages = self.format_inference_data(video_path, prompt, fps, nframes)
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


    # test function on batch size=1
    def collate_fn(self, examples):
        # Get the texts and images, and apply the chat template
        example = examples[0]
        texts = self.processor.apply_chat_template(
            example, 
            tokenize=False, 
            add_generation_prompt=True
        ) # Prepare texts for processing

        img_inp, vid_inp, vid_kwargs = self.process_vision_info(
            example,
            return_video_kwargs=True
        )  # Process the images to extract inputs

        # Tokenize the texts and process the images
        batch = self.processor(
            text=texts, 
            images=img_inp,
            videos=vid_inp,
            padding=True,
            return_tensors="pt",
            **vid_kwargs,
        )  # Encode texts and images into tensors

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(self.processor, Qwen2_5_VLProcessor):  # Check if the processor is Qwen25VLProcessor
            image_tokens = [151652, 151653, 151656]  # Specific image token IDs for Qwen25VLProcessor
        else:
            image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch

        return batch  # Return the prepared batch
