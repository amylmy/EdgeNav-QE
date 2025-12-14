import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, BitsAndBytesConfig
from openvla.training.trainer import VLATrainer
from openvla_early_exit import PrismaticEarlyExit

# Load configuration
with open("qlora_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 1. Define quantization config (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=config["quantization_config"]["load_in_4bit"],
    bnb_4bit_use_double_quant=config["quantization_config"]["bnb_4bit_use_double_quant"],
    bnb_4bit_quant_type=config["quantization_config"]["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, config["quantization_config"]["bnb_4bit_compute_dtype"])
)

# 2. Load quantized early-exit model
model = PrismaticEarlyExit.from_pretrained(
    config["training_config"]["model_name_or_path"],
    torch_dtype=torch.float16,
    device_map=config["training_config"]["device"],
    attn_implementation="eager",
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    exit_layers=[4, 8],  # Define exit layers
    exit_threshold=0.95  # Initial confidence threshold
)

# 3. Apply LoRA (include exit branches in fine-tuning)
lora_config = LoraConfig(
    **config["peft_config"],
    target_modules=["q_proj", "v_proj"] + [f"exit_heads.{l}.weight" for l in [4,8]]  # Add exit branches
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 4. Load datasets (reuse QLoRA logic)
train_dataset = load_dataset("json", data_files=config["training_config"]["train_data_path"])["train"]
val_dataset = load_dataset("json", data_files=config["training_config"]["val_data_path"])["train"]

# 5. Training arguments (reuse QLoRA logic)
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

# Custom loss function: balance main head and exit branch loss
def custom_loss_fn(outputs, labels):
    action_pred = outputs["action_pred"]
    loss = nn.MSELoss()(action_pred, labels)
    # Add supervision for exit branches if early exit is triggered
    if outputs["exit_layer"] in [4,8]:
        exit_head = model.exit_heads[str(outputs["exit_layer"])]
        exit_pred = exit_head(outputs["transformer_outputs"][:,0,:])
        loss += 0.1 * nn.MSELoss()(exit_pred, labels)  # Weighted supervision for exit branches
    return loss

# Override Trainer's compute_loss method
class EarlyExitVLATrainer(VLATrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("action")
        outputs = model(**inputs)
        loss = custom_loss_fn(outputs, labels)
        return (loss, outputs) if return_outputs else loss

# Initialize trainer and start fine-tuning
trainer = EarlyExitVLATrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
model.save_pretrained(config["training_config"]["output_dir"])
print(f"QLoRA + Early Exit fine-tuning completed. Weights saved to {config['training_config']['output_dir']}")