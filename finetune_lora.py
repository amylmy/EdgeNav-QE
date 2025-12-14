import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from openvla.models.prismatic import PrismaticForVisionAndLanguageGeneration
from openvla.training.trainer import VLATrainer

# Load configuration
with open("lora_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load base model (disable FlashAttention2)
model = PrismaticForVisionAndLanguageGeneration.from_pretrained(
    config["training_config"]["model_name_or_path"],
    torch_dtype=torch.float16 if config["training_config"]["fp16"] else torch.float32,
    device_map=config["training_config"]["device"],
    attn_implementation="eager"
)

# Apply LoRA configuration
lora_config = LoraConfig(**config["peft_config"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Print trainable params (~0.1% of total)

# Load datasets
train_dataset = load_dataset("json", data_files=config["training_config"]["train_data_path"])["train"]
val_dataset = load_dataset("json", data_files=config["training_config"]["val_data_path"])["train"]

# Training arguments
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
    report_to="none"  # Disable wandb logging (optional)
)

# Initialize trainer and start fine-tuning
trainer = VLATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Save LoRA weights
model.save_pretrained(config["training_config"]["output_dir"])
print(f"LoRA fine-tuning completed. Weights saved to {config['training_config']['output_dir']}")