import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, BitsAndBytesConfig
from openvla.models.prismatic import PrismaticForVisionAndLanguageGeneration
from openvla.training.trainer import VLATrainer

# Load config
with open("qlora_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 1. Define quantization config (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=config["quantization_config"]["load_in_4bit"],
    bnb_4bit_use_double_quant=config["quantization_config"]["bnb_4bit_use_double_quant"],
    bnb_4bit_quant_type=config["quantization_config"]["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, config["quantization_config"]["bnb_4bit_compute_dtype"])
)

# 2. Load quantized base model
model = PrismaticForVisionAndLanguageGeneration.from_pretrained(
    config["training_config"]["model_name_or_path"],
    torch_dtype=torch.float16,
    device_map=config["training_config"]["device"],
    attn_implementation="eager",
    quantization_config=bnb_config,  # Apply QLoRA quantization
    low_cpu_mem_usage=True
)

# 3. Apply LoRA to quantized model
lora_config = LoraConfig(**config["peft_config"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # ~0.1% of total params (even smaller with 4-bit)

# 4. Load datasets (same as LoRA)
train_dataset = load_dataset("json", data_files=config["training_config"]["train_data_path"])["train"]
val_dataset = load_dataset("json", data_files=config["training_config"]["val_data_path"])["train"]

# 5. Training arguments (optimized for QLoRA)
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

# 6. Start QLoRA fine-tuning
trainer = VLATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
trainer.train()

# Save QLoRA weights
model.save_pretrained(config["training_config"]["output_dir"])
print(f"QLoRA fine-tuning completed. Weights saved to {config['training_config']['output_dir']}")