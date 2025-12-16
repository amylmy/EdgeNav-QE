import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, TrainingArguments
from openvla_early_exit import PrismaticEarlyExit  # Your DEE model
from openvla.training.trainer import VLATrainer

# --------------------------
# 1. Initialize Configs
# --------------------------
# Quantization + LoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="VISION_AND_LANGUAGE_GENERATION",
    target_modules=["q_proj", "v_proj", "exit_heads.4.weight", "exit_heads.8.weight"]
)

# Training args (fixed for data size sweep)
training_args = TrainingArguments(
    output_dir="./qlora_dee_learning_curves",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    num_train_epochs=10,  # Fixed for data size sweep
    logging_steps=10,
    save_strategy="no",
    fp16=True,
    remove_unused_columns=False,
    report_to="none"
)

# --------------------------
# 2. Load Navigation Dataset
# --------------------------
full_dataset = load_dataset("json", data_files="./data/habitat_navigation.jsonl")["train"]
val_dataset = load_dataset("json", data_files="./data/habitat_navigation_val.jsonl")["train"]

# Data size fractions to sweep
data_fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
learning_curve_data = []

# --------------------------
# 3. Incremental Fine-Tuning (Data Size Sweep)
# --------------------------
for frac in tqdm(data_fractions, desc="Training on data fractions"):
    # Subsample training data
    train_subset = full_dataset.shuffle(seed=42).select(range(int(len(full_dataset)*frac)))
    
    # Load DEE model (reset for each fraction)
    model = PrismaticEarlyExit.from_pretrained(
        "../models/openvla-7b",
        torch_dtype=torch.float16,
        device_map="mps",
        quantization_config=bnb_config,
        exit_layers=[4, 8],
        exit_thresholds={"4": 0.80, "8": 0.85}
    )
    model = get_peft_model(model, lora_config)
    
    # Custom Trainer (track SR/EER)
    class LCTrainer(VLATrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.epoch_metrics = []
        
        def compute_metrics(self, eval_pred):
            # Calculate SR (success rate) for navigation
            preds, labels = eval_pred
            success = np.sum(np.linalg.norm(preds - labels, axis=-1) < 0.5)  # <0.5m error = success
            sr = (success / len(preds)) * 100
            
            # Calculate EER (early exit rate) on val set
            eer_4 = np.mean([1 if exit_layer ==4 else 0 for exit_layer in self.model.exit_layer_history])
            eer_8 = np.mean([1 if exit_layer ==8 else 0 for exit_layer in self.model.exit_layer_history])
            total_eer = (eer_4 + eer_8) * 100
            
            return {
                "sr": sr,
                "eer_4": eer_4 * 100,
                "eer_8": eer_8 * 100,
                "total_eer": total_eer,
                "loss": super().compute_loss(self.model, self.eval_dataset).item()
            }
    
    # Initialize trainer
    trainer = LCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_subset,
        eval_dataset=val_dataset
    )
    
    # Train and log metrics
    trainer.train()
    val_metrics = trainer.evaluate()
    
    # Log to learning curve data
    learning_curve_data.append({
        "data_fraction": frac,
        "data_size": len(train_subset),
        "train_sr": trainer.compute_metrics((trainer.predict(train_subset).predictions, train_subset["action"]))["sr"],
        "val_sr": val_metrics["eval_sr"],
        "val_eer_4": val_metrics["eval_eer_4"],
        "val_eer_8": val_metrics["eval_eer_8"],
        "val_total_eer": val_metrics["eval_total_eer"],
        "train_loss": trainer.state.log_history[-1]["loss"],
        "val_loss": val_metrics["eval_loss"],
        "avg_latency": np.mean(trainer.model.latency_history)  # Track inference latency
    })

# --------------------------
# 4. Convert to DataFrame & Plot Learning Curves
# --------------------------
lc_df = pd.DataFrame(learning_curve_data)

# Plot 1: SR vs Training Data Size
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.lineplot(x="data_size", y="train_sr", data=lc_df, marker="o", label="Train SR")
sns.lineplot(x="data_size", y="val_sr", data=lc_df, marker="s", label="Validation SR")
plt.axhline(y=95, color="red", linestyle="--", label="Target SR (95%)")
plt.xlabel("Training Data Size (samples)")
plt.ylabel("Success Rate (SR) %")
plt.title("SR vs Training Data Size (QLoRA+DEE)")
plt.grid(True)
plt.legend()

# Plot 2: EER vs Training Data Size
plt.subplot(2, 2, 2)
sns.lineplot(x="data_size", y="val_eer_4", data=lc_df, marker="o", label="Layer 4 EER")
sns.lineplot(x="data_size", y="val_eer_8", data=lc_df, marker="s", label="Layer 8 EER")
sns.lineplot(x="data_size", y="val_total_eer", data=lc_df, marker="^", label="Total EER")
plt.xlabel("Training Data Size (samples)")
plt.ylabel("Early Exit Rate (EER) %")
plt.title("DEE EER vs Training Data Size")
plt.grid(True)
plt.legend()

# Plot 3: Loss vs Training Data Size (Overfitting Check)
plt.subplot(2, 2, 3)
sns.lineplot(x="data_size", y="train_loss", data=lc_df, marker="o", label="Train Loss")
sns.lineplot(x="data_size", y="val_loss", data=lc_df, marker="s", label="Validation Loss")
plt.xlabel("Training Data Size (samples)")
plt.ylabel("MSE Loss")
plt.title("Loss vs Training Data Size (Overfitting Check)")
plt.grid(True)
plt.legend()

# Plot 4: Latency vs SR (Efficiency-Performance Tradeoff)
plt.subplot(2, 2, 4)
sns.scatterplot(x="val_sr", y="avg_latency", data=lc_df, size="val_total_eer", sizes=(100, 300))
plt.xlabel("Validation SR %")
plt.ylabel("Avg Inference Latency (s/sample)")
plt.title("Latency vs SR (Size = Total EER)")
plt.grid(True)

plt.tight_layout()
plt.savefig("./qlora_dee_learning_curves.png", dpi=300)
plt.show()

# --------------------------
# 5. Key Metrics Summary
# --------------------------
print("=== QLoRA+DEE Learning Curve Summary ===")
print(f"Data Size at SR Plateau: {lc_df[lc_df['val_sr'] >= 95]['data_size'].min()} samples")
print(f"Max Validation SR: {lc_df['val_sr'].max():.1f}% (Data Size: {lc_df.loc[lc_df['val_sr'].idxmax()]['data_size']} samples)")
print(f"Converged EER: {lc_df['val_total_eer'].iloc[-1]:.1f}% (Layer 4: {lc_df['val_eer_4'].iloc[-1]:.1f}%, Layer 8: {lc_df['val_eer_8'].iloc[-1]:.1f}%)")
print(f"Overfitting Gap (Train-Validation SR): {lc_df['train_sr'].iloc[-1] - lc_df['val_sr'].iloc[-1]:.1f}%")