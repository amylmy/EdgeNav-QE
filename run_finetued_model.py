import torch
from PIL import Image
from openvla.models.prismatic import PrismaticProcessor, PrismaticForVisionAndLanguageGeneration
from peft import PeftModel

# Load base model
base_model = PrismaticForVisionAndLanguageGeneration.from_pretrained(
    "../models/openvla-7b",
    torch_dtype=torch.float16,
    device_map="mps",  # Use "cuda" for Linux GPUs
    attn_implementation="eager"
)

# Load LoRA fine-tuned weights
finetuned_model = PeftModel.from_pretrained(
    base_model,
    "./finetuned_openvla_lora"
)

# Inference pipeline
processor = PrismaticProcessor.from_pretrained("../models/openvla-7b")
image = Image.open("test_scene.png").convert("RGB")
inputs = processor(
    images=image,
    text="pick up the cube",
    return_tensors="pt"
).to("mps", torch.float16)  # Match device to training

with torch.no_grad():
    action_pred = finetuned_model.predict_action(**inputs)

# Print predicted action
action_pred = action_pred.detach().cpu().numpy()[0]
print(f"Fine-tuned predicted action: {action_pred.round(6)}")