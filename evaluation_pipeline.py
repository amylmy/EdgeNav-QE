import torch
import pandas as pd
import numpy as np
import time
from peft import PeftModel
from transformers import BitsAndBytesConfig
from openvla_early_exit import PrismaticEarlyExit
from openvla.models.prismatic import PrismaticProcessor

# --------------------------
# 1. Initialize Model/Processor
# --------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load base model (support per-layer thresholds)
base_model = PrismaticEarlyExit.from_pretrained(
    "../models/openvla-7b",
    torch_dtype=torch.float16,
    device_map="mps",  # "cuda" for GPU, "cpu" for CPU
    attn_implementation="eager",
    quantization_config=bnb_config,
    exit_layers=[4, 8],
    exit_threshold=1.0  # Override later per test case
)

# Load QLoRA weights
model = PeftModel.from_pretrained(base_model, "./finetuned_openvla_qlora")
processor = PrismaticProcessor.from_pretrained("../models/openvla-7b")

# --------------------------
# 2. Load Test Dataset
# --------------------------
# Test set with stratified complexity labels (simple/medium/complex)
test_df = pd.read_json("./data/stratified_test_set.jsonl", lines=True)
test_samples = len(test_df)

# --------------------------
# 3. Define Threshold Test Matrix
# --------------------------
threshold_matrix = [
    {"id": "T0", "thresh_4": 1.0, "thresh_8": 1.0},  # Baseline (no EE)
    {"id": "T1", "thresh_4": 0.80, "thresh_8": 0.85},
    {"id": "T2", "thresh_4": 0.85, "thresh_8": 0.90},
    {"id": "T3", "thresh_4": 0.90, "thresh_8": 0.95},
    {"id": "T4", "thresh_4": 0.95, "thresh_8": 0.98},
    {"id": "T5", "thresh_4": 0.88, "thresh_8": 0.93}   # Custom calibrated
]

# --------------------------
# 4. Run Evaluation for Each Threshold Set
# --------------------------
evaluation_results = []

for test_case in threshold_matrix:
    # Set per-layer thresholds (update model to support layer-specific thresholds)
    model.exit_thresholds = {4: test_case["thresh_4"], 8: test_case["thresh_8"]}
    
    # Initialize metrics for this test case
    metrics = {
        "test_id": test_case["id"],
        "thresh_4": test_case["thresh_4"],
        "thresh_8": test_case["thresh_8"],
        "layer_4_exits": 0,
        "layer_8_exits": 0,
        "final_exits": 0,
        "total_latency": 0.0,
        "peak_memory": 0.0,
        "mae_list": [],
        "mae_by_complexity": {"simple": [], "medium": [], "complex": []}
    }
    
    # Run inference on all test samples
    for idx, row in test_df.iterrows():
        # Preprocess input
        image = Image.open(row["image_path"]).convert("RGB")
        inputs = processor(
            images=image,
            text=row["language_instruction"],
            return_tensors="pt"
        ).to("mps", torch.float16)
        
        # Track peak memory (macOS: use torch.cuda.max_memory_allocated for GPU)
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            start_mem = torch.mps.max_memory_allocated() / 1024**2  # Convert to MB
        
        # Track latency
        start_time = time.time()
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            action_pred = outputs["action_pred"]
            exit_layer = outputs["exit_layer"]
            confidence = outputs["confidence"]
        
        # Calculate latency and memory
        latency = time.time() - start_time
        metrics["total_latency"] += latency
        
        if torch.backends.mps.is_available():
            end_mem = torch.mps.max_memory_allocated() / 1024**2
            metrics["peak_memory"] = max(metrics["peak_memory"], end_mem - start_mem)
        
        # Update exit counts
        if exit_layer == 4:
            metrics["layer_4_exits"] += 1
        elif exit_layer == 8:
            metrics["layer_8_exits"] += 1
        else:
            metrics["final_exits"] += 1
        
        # Calculate MAE
        y_true = torch.tensor(row["action"], dtype=torch.float16).to("mps")
        mae = torch.mean(torch.abs(action_pred - y_true)).item()
        metrics["mae_list"].append(mae)
        
        # Group MAE by sample complexity
        complexity = row["complexity_label"]
        metrics["mae_by_complexity"][complexity].append(mae)
    
    # --------------------------
    # 5. Compute Aggregate Metrics
    # --------------------------
    # Efficiency metrics
    metrics["total_samples"] = test_samples
    metrics["early_exit_rate"] = (metrics["layer_4_exits"] + metrics["layer_8_exits"]) / test_samples * 100
    metrics["layer_4_eer"] = metrics["layer_4_exits"] / test_samples * 100
    metrics["layer_8_eer"] = metrics["layer_8_exits"] / test_samples * 100
    metrics["avg_latency"] = metrics["total_latency"] / test_samples
    metrics["throughput"] = test_samples / metrics["total_latency"]
    
    # Accuracy metrics
    metrics["avg_mae"] = np.mean(metrics["mae_list"])
    metrics["rel_mae"] = metrics["avg_mae"] / evaluation_results[0]["avg_mae"] * 100 if metrics["test_id"] != "T0" else 100.0  # Compare to baseline
    metrics["norm_mae"] = metrics["avg_mae"] / 2.0  # Normalize to [-1,1] action range
    metrics["failure_rate"] = len([m for m in metrics["mae_list"] if m > 0.1]) / test_samples * 100
    
    # MAE by complexity
    for comp in ["simple", "medium", "complex"]:
        metrics[f"mae_{comp}"] = np.mean(metrics["mae_by_complexity"][comp])
        metrics[f"failure_rate_{comp}"] = len([m for m in metrics["mae_by_complexity"][comp] if m > 0.1]) / len(metrics["mae_by_complexity"][comp]) * 100
    
    # Tradeoff metric: Accuracy-Efficiency Score (AES)
    metrics["aes"] = (1 - metrics["norm_mae"]) * (metrics["early_exit_rate"] / 100)
    
    # Save results
    evaluation_results.append(metrics)
    print(f"Completed evaluation for test case {test_case['id']}")

# Convert results to DataFrame for analysis
eval_df = pd.DataFrame(evaluation_results)
eval_df.to_csv("./threshold_evaluation_results.csv", index=False)
print("Evaluation completed. Results saved to threshold_evaluation_results.csv")