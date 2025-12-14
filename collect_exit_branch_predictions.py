'''
1. Split a Calibration Dataset
Use a held-out validation subset (10–20% of the full validation set) that is representative of task complexity (includes simple/medium/complex samples). For OpenVLA-7B:
Simple samples: Clear images + short instructions (e.g., "pick up red cube")
Complex samples: Occluded images + long instructions (e.g., "stack 3 cubes in blue tray")
2. Collect Exit Branch Predictions
Run the pre-finetuned QLoRA+Early Exit model on the calibration dataset, recording:
For each sample: Predictions from all exit layers (4/8) and the final layer
Ground-truth 7D action vectors
Confidence scores for each exit layer
Inference latency per sample (per layer)
'''
import torch
import pandas as pd
from peft import PeftModel
from transformers import BitsAndBytesConfig
from openvla_early_exit import PrismaticEarlyExit
from openvla.models.prismatic import PrismaticProcessor

# Load model and processor
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = PrismaticEarlyExit.from_pretrained(
    "../models/openvla-7b",
    torch_dtype=torch.float16,
    device_map="mps",
    attn_implementation="eager",
    quantization_config=bnb_config,
    exit_layers=[4, 8],
    exit_threshold=1.0  # Disable early exit to collect all layer outputs
)
model = PeftModel.from_pretrained(base_model, "./finetuned_openvla_qlora")
processor = PrismaticProcessor.from_pretrained("../models/openvla-7b")

# Calibration dataset (replace with your path)
calib_data = pd.read_json("./data/calibration_set.jsonl", lines=True)
results = []

# Collect predictions and confidence scores
for idx, row in calib_data.iterrows():
    # Preprocess input
    image = Image.open(row["image_path"]).convert("RGB")
    inputs = processor(
        images=image,
        text=row["language_instruction"],
        return_tensors="pt"
    ).to("mps", torch.float16)
    
    with torch.no_grad():
        # Forward pass (get all layer outputs)
        layer_4_out = model.get_layer_output(4, **inputs)  # Add this method to PrismaticEarlyExit
        layer_8_out = model.get_layer_output(8, **inputs)
        final_out = model(**inputs)["action_pred"]
        
        # Calculate confidence for each layer
        conf_4 = 1 - torch.var(layer_4_out, dim=-1).mean().item()
        conf_8 = 1 - torch.var(layer_8_out, dim=-1).mean().item()
        conf_final = 1 - torch.var(final_out, dim=-1).mean().item()
        
        # Calculate MAE (error vs ground truth)
        mae_4 = torch.mean(torch.abs(layer_4_out - torch.tensor(row["action"]).to("mps"))).item()
        mae_8 = torch.mean(torch.abs(layer_8_out - torch.tensor(row["action"]).to("mps"))).item()
        mae_final = torch.mean(torch.abs(final_out - torch.tensor(row["action"]).to("mps"))).item()
        
        # Record latency (per layer)
        import time
        start = time.time()
        model.get_layer_output(4, **inputs)
        latency_4 = time.time() - start
        
        start = time.time()
        model.get_layer_output(8, **inputs)
        latency_8 = time.time() - start
        
        start = time.time()
        model(**inputs)
        latency_final = time.time() - start
        
        # Save results
        results.append({
            "sample_idx": idx,
            "conf_4": conf_4,
            "conf_8": conf_8,
            "mae_4": mae_4,
            "mae_8": mae_8,
            "mae_final": mae_final,
            "latency_4": latency_4,
            "latency_8": latency_8,
            "latency_final": latency_final,
            "complexity": row["complexity_label"]  # Manual label: simple/medium/complex
        })

# Save calibration results
calib_df = pd.DataFrame(results)
calib_df.to_csv("./calibration_results.csv", index=False)