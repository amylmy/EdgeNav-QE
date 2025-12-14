import torch
from PIL import Image
from peft import PeftModel
from transformers import BitsAndBytesConfig
from openvla_early_exit import PrismaticEarlyExit
from openvla.models.prismatic import PrismaticProcessor

# 1. Quantization config (match training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 2. Load quantized early-exit base model
base_model = PrismaticEarlyExit.from_pretrained(
    "../models/openvla-7b",
    torch_dtype=torch.float16,
    device_map="mps",
    attn_implementation="eager",
    quantization_config=bnb_config,
    exit_layers=[4, 8],
    exit_threshold=0.95  # Adjustable based on scenario complexity
)

# 3. Load QLoRA + Early Exit weights
finetuned_model = PeftModel.from_pretrained(
    base_model,
    "./finetuned_openvla_qlora"
)

# 4. Inference with dynamic early exit
processor = PrismaticProcessor.from_pretrained("../models/openvla-7b")
image = Image.open("test_scene.png").convert("RGB")
inputs = processor(
    images=image,
    text="pick up the red cube",
    return_tensors="pt"
).to("mps", torch.float16)

with torch.no_grad():
    outputs = finetuned_model(**inputs)
    action_pred = outputs["action_pred"]
    exit_layer = outputs["exit_layer"]
    confidence = outputs["confidence"]

# Post-process and print results
action_pred = action_pred.detach().cpu().numpy()[0]
print(f"Predicted Action: {action_pred.round(6)}")
print(f"Early Exit at Layer: {exit_layer} (Confidence: {confidence.item():.2f})")