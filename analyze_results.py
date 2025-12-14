import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# --------------------------
# 1. Load Evaluation Results
# --------------------------
eval_df = pd.read_csv("./threshold_evaluation_results.csv")
baseline = eval_df[eval_df["test_id"] == "T0"].iloc[0]  # No early exit baseline

# --------------------------
# 2. Key Visualizations
# --------------------------
plt.rcParams["figure.figsize"] = (15, 10)

# Plot 1: Latency vs MAE (Pareto Front)
plt.subplot(2, 3, 1)
# Plot test cases
plt.scatter(eval_df["avg_latency"], eval_df["avg_mae"], c=eval_df["early_exit_rate"], cmap="viridis", s=100)
# Highlight baseline (T0)
plt.scatter(baseline["avg_latency"], baseline["avg_mae"], marker="*", color="red", s=200, label="Baseline (No EE)")
# Add labels for test IDs
for idx, row in eval_df.iterrows():
    plt.annotate(row["test_id"], (row["avg_latency"], row["avg_mae"]), xytext=(5,5), textcoords="offset points")

plt.xlabel("Average Latency (s/sample)")
plt.ylabel("Average MAE")
plt.title("Latency vs MAE (Color = Early Exit Rate %)")
plt.colorbar(label="Early Exit Rate (%)")
plt.grid(True)
plt.legend()

# Plot 2: Early Exit Rate vs Relative MAE
plt.subplot(2, 3, 2)
plt.plot(eval_df["early_exit_rate"], eval_df["rel_mae"], marker="o", linewidth=2, markersize=8)
plt.axhline(y=110, color="r", linestyle="--", label="Max Acceptable R-MAE (110%)")
plt.xlabel("Early Exit Rate (%)")
plt.ylabel("Relative MAE vs Baseline (%)")
plt.title("Early Exit Rate vs Relative MAE")
plt.grid(True)
plt.legend()

# Plot 3: AES (Accuracy-Efficiency Score) Across Test Cases
plt.subplot(2, 3, 3)
sns.barplot(x="test_id", y="aes", data=eval_df)
plt.xlabel("Test Case (Threshold Set)")
plt.ylabel("Accuracy-Efficiency Score (AES)")
plt.title("AES Across Threshold Sets (Higher = Better)")
plt.grid(True, axis="y")

# Plot 4: MAE by Complexity
plt.subplot(2, 3, 4)
comp_metrics = ["mae_simple", "mae_medium", "mae_complex"]
x = np.arange(len(eval_df["test_id"]))
width = 0.25

for i, comp in enumerate(comp_metrics):
    plt.bar(x + i*width, eval_df[comp], width, label=comp.replace("mae_", "").capitalize())

plt.xlabel("Test Case")
plt.ylabel("MAE")
plt.title("MAE by Sample Complexity")
plt.xticks(x + width, eval_df["test_id"])
plt.legend()
plt.grid(True, axis="y")

# Plot 5: Failure Rate vs Threshold
plt.subplot(2, 3, 5)
plt.plot(eval_df["thresh_4"], eval_df["failure_rate"], marker="o", label="Layer 4 Threshold vs Failure Rate")
plt.plot(eval_df["thresh_8"], eval_df["failure_rate"], marker="s", label="Layer 8 Threshold vs Failure Rate")
plt.xlabel("Confidence Threshold")
plt.ylabel("Failure Rate (%)")
plt.title("Threshold vs Failure Rate")
plt.legend()
plt.grid(True)

# Plot 6: Throughput vs Normalized MAE
plt.subplot(2, 3, 6)
plt.scatter(eval_df["norm_mae"], eval_df["throughput"], c=eval_df["test_id"], cmap="tab10", s=100)
plt.xlabel("Normalized MAE")
plt.ylabel("Throughput (samples/sec)")
plt.title("Normalized MAE vs Throughput")
plt.grid(True)

plt.tight_layout()
plt.savefig("./threshold_impact_analysis.png", dpi=300)
plt.show()

# --------------------------
# 3. Statistical Validation
# --------------------------
# 3.1 Correlation Analysis (Threshold vs Metrics)
corr_matrix = eval_df[["thresh_4", "thresh_8", "early_exit_rate", "avg_latency", "avg_mae", "aes"]].corr()
print("=== Correlation Matrix (Threshold vs Key Metrics) ===")
print(corr_matrix)

# 3.2 Statistical Significance (vs Baseline)
# Test if MAE/latency differences are statistically significant (t-test)
for test_id in eval_df["test_id"].unique():
    if test_id == "T0":
        continue
    test_data = eval_df[eval_df["test_id"] == test_id].iloc[0]
    
    # Compare MAE to baseline (two-tailed t-test)
    mae_test = eval_df[eval_df["test_id"] == test_id]["avg_mae"].values[0]
    mae_baseline = baseline["avg_mae"]
    # Assume 5 runs for statistical power (replace with your replicate data)
    mae_replicates = [mae_test * np.random.normal(1, 0.01) for _ in range(5)]
    baseline_replicates = [mae_baseline * np.random.normal(1, 0.01) for _ in range(5)]
    
    t_stat, p_val = stats.ttest_ind(mae_replicates, baseline_replicates)
    sig = "Significant" if p_val < 0.05 else "Not Significant"
    
    print(f"\n=== Test Case {test_id} vs Baseline ===")
    print(f"MAE: {mae_test:.4f} (Baseline: {mae_baseline:.4f}) | p-value: {p_val:.4f} ({sig})")
    print(f"Latency Reduction: {100*(1 - test_data['avg_latency']/baseline['avg_latency']):.2f}%")
    print(f"Early Exit Rate: {test_data['early_exit_rate']:.2f}%")
    print(f"AES: {test_data['aes']:.4f} (Baseline: {baseline['aes']:.4f})")

# --------------------------
# 4. Optimal Threshold Selection
# --------------------------
# Select threshold set with highest AES, MAE ≤ 110% of baseline, and latency reduction ≥ 20%
optimal_candidates = eval_df[
    (eval_df["rel_mae"] ≤ 110) & 
    (eval_df["avg_latency"] ≤ 0.8 * baseline["avg_latency"])  # ≥20% latency reduction
]

if not optimal_candidates.empty:
    optimal_threshold = optimal_candidates.loc[optimal_candidates["aes"].idxmax()]
    print("\n=== Optimal Threshold Set ===")
    print(f"Test ID: {optimal_threshold['test_id']}")
    print(f"Layer 4 Threshold: {optimal_threshold['thresh_4']:.2f}")
    print(f"Layer 8 Threshold: {optimal_threshold['thresh_8']:.2f}")
    print(f"Key Metrics: EER={optimal_threshold['early_exit_rate']:.2f}%, MAE={optimal_threshold['avg_mae']:.4f}, Latency={optimal_threshold['avg_latency']:.4f}s, AES={optimal_threshold['aes']:.4f}")
else:
    print("\nNo optimal threshold set found (meets accuracy/latency targets). Adjust targets or thresholds.")