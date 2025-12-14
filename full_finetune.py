import torch
import yaml
from datasets import load_dataset
from transformers import TrainingArguments
from openvla.models.prismatic import PrismaticForVisionAndLanguageGeneration
from openvla.training.trainer import VLATrainer

# Load config (same as LoRA script)
with open("full_finetune_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load base model (enable FlashAttention2 for GPUs)
model = PrismaticForVisionAndLanguageGeneration.from_pretrained(
    config["training_config"]["model_name_or_path"],
    torch_dtype=torch.float16,
    device_map="auto",  # Auto-allocate GPU VRAM
    attn_implementation="flash_attention_2"
)

# Load datasets (same as LoRA script)
train_dataset = load_dataset("json", data_files=config["training_config"]["train_data_path"])["train"]
val_dataset = load_dataset("json", data_files=config["training_config"]["val_data_path"])["train"]

# Training arguments (same as LoRA script)
training_args = TrainingArguments(
    output_dir=config["training_config"]["output_dir"],
    per_device_train_batch_size=config["training_config"]["per_device_train_batch_size"],
    gradient_accumulation_steps=config["training_config"]["gradient_accumulation_steps"],
    learning_rate=config["training_config"]["learning_rate"],
    num_train_epochs=config["training_config"]["num_train_epochs"],
    logging_steps=config["training_config"]["logging_steps"],
    save_steps=config["training_config"]["save_steps"],
    fp16=config["training_config"]["fp16"],
    remove_unused_columns=False,
    report_to="none"
)

# Train without LoRA wrapping
trainer = VLATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
trainer.train()

# Save full fine-tuned weights
model.save_pretrained(config["training_config"]["output_dir"])
print(f"Full fine-tuning completed. Weights saved to {config['training_config']['output_dir']}")