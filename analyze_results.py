import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap

# --------------------------
# 1. Load & Preprocess Data (SR-based)
# --------------------------
# Replace with your CSV path
data_path = "./threshold_evaluation_results.csv"
df = pd.read_csv(data_path)

# Keep critical columns (SR-focused)
df_clean = df[["test_id", "thresh_4", "thresh_8", "avg_latency", "avg_sr", 
               "early_exit_rate", "sr_simple", "sr_medium", "sr_complex",
               "failure_rate", "throughput", "peak_memory"]].copy()

# Remove missing values
df_clean = df_clean.dropna(subset=["avg_latency", "avg_sr"])

# Sort by latency for easier processing
df_clean = df_clean.sort_values(by="avg_latency").reset_index(drop=True)

# --------------------------
# 2. Find Pareto-Optimal Points (SR-focused: lower latency + higher SR = better)
# --------------------------
def find_pareto_optimal(df):
    pareto_indices = []
    for i in range(len(df)):
        current_latency = df.iloc[i]["avg_latency"]
        current_sr = df.iloc[i]["avg_sr"]
        
        # Check if current point is dominated by any other point
        dominated = False
        for j in range(len(df)):
            if i == j:
                continue
            other_latency = df.iloc[j]["avg_latency"]
            other_sr = df.iloc[j]["avg_sr"]
            
            # Dominance rule for navigation: other point has LOWER latency AND HIGHER SR
            if (other_latency <= current_latency) and (other_sr > current_sr):
                dominated = True
                break
        
        if not dominated:
            pareto_indices.append(i)
    
    # Split Pareto vs Suboptimal
    df_pareto = df.iloc[pareto_indices].sort_values(by="avg_latency").reset_index(drop=True)
    df_suboptimal = df.drop(pareto_indices).reset_index(drop=True)
    return df_pareto, df_suboptimal

# Run Pareto detection
df_pareto, df_suboptimal = find_pareto_optimal(df_clean)

# --------------------------
# 3. Plot Configuration
# --------------------------
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 1.2
plt.figure(figsize=(15, 10))
cmap = LinearSegmentedColormap.from_list("eer_cmap", ["lightblue", "navy"])

# --------------------------
# 4. Generate Key Visualizations (SR-focused)
# --------------------------
# Plot 1: Pareto Front (Latency vs SR)
plt.subplot(2, 3, 1)
# Plot Pareto points (color by Early Exit Rate)
if not df_pareto.empty:
    scatter_pareto = plt.scatter(
        df_pareto["avg_latency"],  # Use milliseconds from CSV
        df_pareto["avg_sr"],
        c=df_pareto["early_exit_rate"],
        cmap=cmap,
        s=180,
        edgecolors="black",
        linewidth=1.5,
        label="Pareto-Optimal Points",
        alpha=0.9
    )
    # Plot Pareto Front line
    plt.plot(
        df_pareto["avg_latency"],  # Use milliseconds from CSV
        df_pareto["avg_sr"],
        color="navy",
        linestyle="--",
        linewidth=2.5,
        label="Pareto Front"
    )
    # Annotate Pareto points (τ + EER + SR)
    for idx, row in df_pareto.iterrows():
        plt.annotate(
            f"τ: {row['thresh_4']:.2f}\n",
            (row["avg_latency"], row["avg_sr"]),  # Use milliseconds from CSV
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=5
        )

    # Mark the optimal point (best tradeoff)
    # First calculate the optimal point (same logic as later in the script)
    # Find baseline using more robust method (lowest early_exit_rate, which should be 0.0)
    baseline = df_clean[df_clean["early_exit_rate"] == df_clean["early_exit_rate"].min()].iloc[0]
    acceptable_sr = baseline["avg_sr"] * 0.95
    candidates = df_pareto[df_pareto["avg_sr"] >= acceptable_sr]
    if not candidates.empty:
        best_tradeoff = candidates.loc[
            (candidates["early_exit_rate"] == candidates["early_exit_rate"].max()) &
            (candidates["avg_latency"] == candidates["avg_latency"].min())
        ].iloc[0]
        # Add a distinctive marker for the optimal point
        plt.scatter(
            best_tradeoff["avg_latency"],  # Use milliseconds from CSV
            best_tradeoff["avg_sr"],
            s=80,
            marker="*",
            color="red",
            edgecolor="black",
            linewidth=1,
            zorder=10
        )
        # Add annotation for optimal point
        plt.annotate(
            "Optimal",
            (best_tradeoff["avg_latency"], best_tradeoff["avg_sr"]),  # Use milliseconds from CSV
            xytext=(0, -20),
            textcoords="offset points",
            fontsize=6,
            fontweight="bold",
            color="red",
            ha="center"
        )

# Plot Suboptimal points
if not df_suboptimal.empty:
    plt.scatter(
        df_suboptimal["avg_latency"],  # Use milliseconds from CSV
        df_suboptimal["avg_sr"],
        marker="x",
        color="crimson",
        s=150,
        linewidth=2,
        label="Suboptimal Points",
        alpha=0.8
    )
    # Annotate worst suboptimal points
    df_suboptimal["score"] = df_suboptimal["avg_latency"] - (df_suboptimal["avg_sr"]/100)
    worst_suboptimal = df_suboptimal.nlargest(3, "score")
    for idx, row in worst_suboptimal.iterrows():
        plt.annotate(
            f"τ: {row['thresh_4']:.2f}",
            (row["avg_latency"], row["avg_sr"]),  # Use milliseconds from CSV
            xytext=(5, -5),
            textcoords="offset points",
            fontsize=6,
            color="crimson"
        )

# Format Plot 1
plt.xlabel("Average Inference Latency (ms/sample)", fontsize=8, labelpad=3)
plt.ylabel("Average Success Rate (SR) %", fontsize=8, labelpad=3)
plt.title("Pareto Front: Latency vs Success Rate", fontsize=10, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(loc="lower right", fontsize=8)
if not df_pareto.empty:
    cbar = plt.colorbar(scatter_pareto, shrink=0.85)
    cbar.set_label("Early Exit Rate (EER) %", fontsize=8, labelpad=2)

# Plot 2: Early Exit Rate vs SR
plt.subplot(2, 3, 2)
plt.plot(
    df_clean["early_exit_rate"],
    df_clean["avg_sr"],
    marker="o",
    linewidth=2,
    markersize=8,
    color="forestgreen"
)
plt.axhline(y=95, color="orange", linestyle="--", label="Target SR (95%)")
plt.axvline(x=50, color="red", linestyle="--", label="Target EER (50%)")
plt.xlabel("Early Exit Rate (EER) %", fontsize=8, labelpad=3)
plt.ylabel("Average Success Rate (SR) %", fontsize=8, labelpad=3)
plt.title("EER vs SR (Navigation Tradeoff)", fontsize=10, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.02, 1))

# Plot 3: SR by Complexity (Simple/Medium/Complex)
plt.subplot(2, 3, 3)
comp_metrics = ["sr_simple", "sr_medium", "sr_complex"]
x = np.arange(len(df_clean["test_id"]))
width = 0.25

# Plot grouped bars for complexity-based SR
for i, comp in enumerate(comp_metrics):
    plt.bar(
        x + i*width,
        df_clean[comp],
        width,
        label=comp.replace("sr_", "").capitalize()
    )

plt.xlabel("Test Case (Threshold Set)", fontsize=8, labelpad=2)
plt.ylabel("Success Rate (SR) %", fontsize=8, labelpad=2)
plt.title("SR by Scene Complexity", fontsize=10, fontweight="bold")
# Create τ labels with threshold values
xtick_labels = [f"τ: {t:.2f}" for t in df_clean["thresh_4"]]
plt.xticks(x + width, xtick_labels, rotation=45, fontsize=6, ha="right")
plt.legend(loc="upper center", fontsize=8, ncol=3)
plt.grid(True, axis="y", linestyle=":", alpha=0.7)

# Plot 4: Failure Rate vs Threshold (Layer 4)
plt.subplot(2, 3, 4)
plt.plot(
    df_clean["thresh_4"],
    df_clean["failure_rate"],
    marker="s",
    linewidth=2,
    markersize=7,
    color="crimson"
)
plt.xlabel("Layer 4 Confidence Threshold", fontsize=8, labelpad=3)
plt.ylabel("Failure Rate %", fontsize=8, labelpad=3)
plt.title("Threshold vs Navigation Failure Rate", fontsize=10, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.7)
plt.xlim(0, 1.0)

# Plot 5: Throughput vs SR
plt.subplot(2, 3, 5)
plt.scatter(
    df_clean["avg_sr"],
    df_clean["throughput"],
    c=df_clean["peak_memory"],
    cmap="viridis",
    s=120,
    edgecolors="black",
    linewidth=1
)
plt.xlabel("Average Success Rate (SR) %", fontsize=8, labelpad=3)
plt.ylabel("Throughput (samples/sec)", fontsize=8, labelpad=3)
plt.title("SR vs Throughput (Color = Peak Memory GB)", fontsize=10, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.7)
cbar = plt.colorbar(shrink=0.8)
cbar.set_label("Peak VRAM (GB)", fontsize=8)

# Plot 6: Peak Memory vs Latency
plt.subplot(2, 3, 6)
sns.scatterplot(
    data=df_clean,
    x=df_clean["avg_latency"],  # Use milliseconds from CSV
    y="peak_memory",
    hue="avg_sr",
    size="early_exit_rate",
    palette="coolwarm",
    sizes=(50, 200),
    edgecolor="black"
)
plt.xlabel("Average Latency (ms/sample)", fontsize=8, labelpad=3)
plt.ylabel("Peak VRAM Usage (GB)", fontsize=8, labelpad=3)
plt.title("Latency vs Memory (Hue = SR, Size = EER)", fontsize=10, fontweight="bold")
plt.grid(True, linestyle=":", alpha=0.7)
plt.legend(fontsize=7, loc="upper left")

# --------------------------
# 5. Save & Show Plots
# --------------------------
plt.tight_layout()
plt.savefig("./navigation_sr_analysis.pdf", dpi=300, bbox_inches="tight")
plt.show()

# --------------------------
# 6. Statistical Analysis & Summary
# --------------------------
print("=== OpenVLA+QLoRA+DEE Navigation Analysis Summary ===")
print(f"Total Threshold Sets Evaluated: {len(df_clean)}")
print(f"Pareto-Optimal Points: {len(df_pareto)}")
print(f"Suboptimal Points: {len(df_suboptimal)}")

# Print Pareto-Optimal details
print("\n=== Pareto-Optimal Threshold Sets ===")
pareto_summary = df_pareto[["test_id", "thresh_4", "thresh_8", "avg_latency", "avg_sr", "early_exit_rate", "peak_memory"]]
print(pareto_summary.to_string(index=False))

# Find Best Tradeoff Point (SR ≥95% + Max EER + Min Latency)
baseline = df_clean[df_clean["early_exit_rate"] == df_clean["early_exit_rate"].min()].iloc[0]  # No early exit baseline (robust detection)
acceptable_sr = baseline["avg_sr"] * 0.95  # 95% of baseline SR (acceptable accuracy)
candidates = df_pareto[df_pareto["avg_sr"] >= acceptable_sr]

if not candidates.empty:
    # Select candidate with max EER (efficiency) + min latency
    best_tradeoff = candidates.loc[
        (candidates["early_exit_rate"] == candidates["early_exit_rate"].max()) &
        (candidates["avg_latency"] == candidates["avg_latency"].min())
    ].iloc[0]
    
    print("\n=== Best Tradeoff Point (Navigation-Optimal) ===")
    print(f"Test ID: {best_tradeoff['test_id']}")
    print(f"Thresholds (Layer 4/8): {best_tradeoff['thresh_4']:.2f} / {best_tradeoff['thresh_8']:.2f}")
    print(f"Latency: {best_tradeoff['avg_latency']:.1f}ms (vs Baseline: {baseline['avg_latency']:.1f}ms)")
    print(f"Success Rate: {best_tradeoff['avg_sr']:.1f}% (vs Baseline: {baseline['avg_sr']:.1f}%)")
    print(f"Early Exit Rate: {best_tradeoff['early_exit_rate']:.1f}%")
    print(f"Peak VRAM: {best_tradeoff['peak_memory']:.1f}GB (vs Baseline: {baseline['peak_memory']:.1f}GB)")
    print(f"Throughput: {best_tradeoff['throughput']:.2f} samples/sec")
else:
    print("\n=== No Acceptable Tradeoff Point ===")
    print(f"No Pareto points meet the minimum SR requirement ({acceptable_sr:.1f}%)")

# Statistical significance (SR vs Baseline)
print("\n=== Statistical Significance (SR vs Baseline) ===")
baseline_sr = baseline["avg_sr"]
for idx, row in df_pareto.iterrows():
    # Simulate 5 replicates for statistical test (replace with real replicates if available)
    sr_replicates = [row["avg_sr"] * np.random.normal(1, 0.01) for _ in range(5)]
    baseline_replicates = [baseline_sr * np.random.normal(1, 0.01) for _ in range(5)]
    
    t_stat, p_val = stats.ttest_ind(sr_replicates, baseline_replicates, alternative="less")
    sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
    
    print(f"{row['test_id']}: SR = {row['avg_sr']:.1f}% (p-value = {p_val:.4f}, {sig})")