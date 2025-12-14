'''
Validate Thresholds
Test the calibrated thresholds on a holdout test set to confirm:
Accuracy: MAE of early-exit samples meets the target (≤10% error vs final layer)
Efficiency: Early exit rate (percentage of samples exiting at layer 4/8) meets latency goals
Robustness: Thresholds work for all complexity levels (simple/medium/complex)
'''

# Load test set
test_df = pd.read_json("./data/test_set.jsonl", lines=True)
exit_stats = {"layer_4": 0, "layer_8": 0, "final": 0}
mae_list = []

# Set calibrated thresholds
model.exit_thresholds = {4: optimal_thresh_4, 8: optimal_thresh_8}  # Update model to support per-layer thresholds

for idx, row in test_df.iterrows():
    # Preprocess and infer
    image = Image.open(row["image_path"]).convert("RGB")
    inputs = processor(images=image, text=row["language_instruction"], return_tensors="pt").to("mps", torch.float16)
    
    with torch.no_grad():
        outputs = model(**inputs)
        exit_layer = outputs["exit_layer"]
        action_pred = outputs["action_pred"]
        
        # Update exit stats
        if exit_layer == 4:
            exit_stats["layer_4"] += 1
        elif exit_layer == 8:
            exit_stats["layer_8"] += 1
        else:
            exit_stats["final"] += 1
        
        # Calculate MAE
        mae = torch.mean(torch.abs(action_pred - torch.tensor(row["action"]).to("mps"))).item()
        mae_list.append(mae)

# Print validation results
print(f"Early Exit Stats: Layer 4={exit_stats['layer_4']}, Layer 8={exit_stats['layer_8']}, Final={exit_stats['final']}")
print(f"Overall MAE: {np.mean(mae_list):.4f}")
print(f"Early Exit Rate: {100*(exit_stats['layer_4'] + exit_stats['layer_8'])/len(test_df):.2f}%")
print(f"Average Latency Reduction: {100*(1 - (exit_stats['layer_4']*calib_df['latency_4'].mean() + exit_stats['layer_8']*calib_df['latency_8'].mean() + exit_stats['final']*calib_df['latency_final'].mean())/(len(test_df)*calib_df['latency_final'].mean())):.2f}%")