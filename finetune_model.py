import argparse
import yaml
from loguru import logger
import torch
import pathlib
import wandb

from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

from dataset import UCF_Crime

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
    EVAL_META_PATH = cfg['eval_meta_path']
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

    # Load dataset
    train_ds = UCF_Crime(
        pathlib.Path(DATASET_PATH), 
        META_PATH,
    )
    eval_ds = UCF_Crime(
        pathlib.Path(DATASET_PATH), 
        EVAL_META_PATH,
    )
    logger.success("dataset initialised")

    # Load quantized model
    # Setup quantized config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    backend = Model(
        model_id=MODEL_ID, 
        cache_dir=CACHE_DIR, 
        quantization_config=bnb_config
    )

    logger.success(f"model loaded with quantization config {bnb_config}")

    # Print number of parameters
    s = 0
    for n, p in backend.model.named_parameters():
        s += p.numel()
    print(f"Model {MODEL} has total {s} parameters")

    # Setup LORA
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    # Apply PEFT model adaptation
    peft_model = get_peft_model(
        backend.model, 
        peft_config
    )

    # Print trainable parameters
    peft_model.print_trainable_parameters()

    # Setup training arguments
    training_args = SFTConfig(
        output_dir="qwen25-3b-instruct-trl-sft-ChartQA",  # Directory to save the model
        num_train_epochs=3,  # Number of training epochs
        per_device_train_batch_size=4,  # Batch size for training
        per_device_eval_batch_size=4,  # Batch size for evaluation
        gradient_accumulation_steps=8,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=2e-4,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=10,  # Steps interval for logging
        eval_steps=10,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=20,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=1024  # Maximum sequence length for input
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    logger.success("Training arguments initialised")

    # Initialize wandb
    wandb.init(
        project="VLM-deviant-behavior-research",  # change this
        name=f"{MODEL.lower()}-trl-sft-ChartQA",  # change this
        config=training_args,
    )

    # Format data
    train_set = [backend.format_data(train_ds[idx]['path'], cfg['prompt'], cfg['fps']) for idx in range(len(train_ds))]
    eval_set = [backend.format_data(eval_ds[idx]['path'], cfg['prompt'], cfg['fps']) for idx in range(10)]

    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=backend.model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=backend.collate_fn,
        peft_config=peft_config,
        processing_class=backend.processor.tokenizer,
    )
    logger.success("SFT trainer initialized")

    # Train model
    logger.info(f"Training model {MODEL}")
    trainer.train()

    # # Save model
    # print(f"Saving model {MODEL}")
    # trainer.save_model(f"finetuned_{MODEL_ID}") 
