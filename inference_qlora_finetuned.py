import torch
from PIL import Image
from peft import PeftModel
from transformers import BitsAndBytesConfig
from openvla.models.prismatic import PrismaticProcessor, PrismaticForVisionAndLanguageGeneration

# 1. Quantization config (match training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# 2. Load quantized base model
base_model = PrismaticForVisionAndLanguageGeneration.from_pretrained(
    "../models/openvla-7b",
    torch_dtype=torch.float16,
    device_map="mps",
    attn_implementation="eager",
    quantization_config=bnb_config
)

# 3. Load QLoRA fine-tuned weights
finetuned_model = PeftModel.from_pretrained(
    base_model,
    "./finetuned_openvla_qlora"
)

# 4. Inference
processor = PrismaticProcessor.from_pretrained("../models/openvla-7b")
image = Image.open("test_scene.png").convert("RGB")
inputs = processor(
    images=image,
    text="pick up the red cube",
    return_tensors="pt"
).to("mps", torch.float16)

with torch.no_grad():
    action_pred = finetuned_model.predict_action(**inputs)

action_pred = action_pred.detach().cpu().numpy()[0]
print(f"QLoRA Predicted Action: {action_pred.round(6)}")