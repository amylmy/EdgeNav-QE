'''
1. Define Accuracy Targets
First set a minimum acceptable accuracy (MAE) for early exit branches:
Example: MAE of exit layer 4 ≤ 1.1x the final layer’s MAE (10% error tolerance)
Example: MAE of exit layer 8 ≤ 1.05x the final layer’s MAE (5% error tolerance)
2. Calculate Thresholds via Precision-Recall Curves
For each exit layer, plot confidence thresholds vs. valid early exit rate 
(samples with MAE ≤ target) and select the threshold that maximizes early exits 
while meeting accuracy targets:
'''
import matplotlib.pyplot as plt
import numpy as np

# Load calibration results
calib_df = pd.read_csv("./calibration_results.csv")

# Define accuracy targets (10% tolerance for layer 4, 5% for layer 8)
target_mae_4 = calib_df["mae_final"].mean() * 1.1
target_mae_8 = calib_df["mae_final"].mean() * 1.05

# Step 1: Calibrate threshold for layer 4
thresholds_4 = np.linspace(0.8, 0.99, 20)
valid_exit_rate_4 = []
for t in thresholds_4:
    # Valid samples: conf ≥ t AND mae_4 ≤ target_mae_4
    valid = calib_df[(calib_df["conf_4"] >= t) & (calib_df["mae_4"] <= target_mae_4)]
    rate = len(valid) / len(calib_df)
    valid_exit_rate_4.append(rate)

# Step 2: Calibrate threshold for layer 8
thresholds_8 = np.linspace(0.85, 0.99, 20)
valid_exit_rate_8 = []
for t in thresholds_8:
    valid = calib_df[(calib_df["conf_8"] >= t) & (calib_df["mae_8"] <= target_mae_8)]
    rate = len(valid) / len(calib_df)
    valid_exit_rate_8.append(rate)

# Step 3: Plot and select optimal thresholds
plt.figure(figsize=(12, 4))

# Plot layer 4
plt.subplot(1, 2, 1)
plt.plot(thresholds_4, valid_exit_rate_4, marker="o")
plt.axhline(y=0.4, color="r", linestyle="--", label="40% exit rate target")  # Example target
plt.xlabel("Layer 4 Confidence Threshold")
plt.ylabel("Valid Early Exit Rate")
plt.title("Layer 4: Threshold vs Valid Exit Rate")
plt.legend()
plt.grid(True)

# Plot layer 8
plt.subplot(1, 2, 2)
plt.plot(thresholds_8, valid_exit_rate_8, marker="o", color="orange")
plt.axhline(y=0.7, color="r", linestyle="--", label="70% exit rate target")  # Example target
plt.xlabel("Layer 8 Confidence Threshold")
plt.ylabel("Valid Early Exit Rate")
plt.title("Layer 8: Threshold vs Valid Exit Rate")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("./threshold_calibration.png")

# Select optimal thresholds (example values)
optimal_thresh_4 = thresholds_4[np.argmax([r for r in valid_exit_rate_4 if r >= 0.4])]
optimal_thresh_8 = thresholds_8[np.argmax([r for r in valid_exit_rate_8 if r >= 0.7])]
print(f"Optimal threshold for layer 4: {optimal_thresh_4:.2f}")
print(f"Optimal threshold for layer 8: {optimal_thresh_8:.2f}")