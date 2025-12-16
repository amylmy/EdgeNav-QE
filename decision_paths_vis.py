import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as mpatches

# --------------------------
# 1. Configure Plot Style & Global Parameters
# --------------------------
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

# Navigation parameters (3 phases: 1-15, 16-25, 26-35)
timesteps = np.arange(1, 36)  # 35 timesteps total
phase1 = timesteps[:15]  # 1-15: Open Hallway
phase2 = timesteps[15:25] # 16-25: Obstacle Avoidance
phase3 = timesteps[25:]  # 26-35: Recovery

# --------------------------
# 2. Simulate Realistic Data for Entropy & Computational Depth
# --------------------------
# Entropy H(p): low in phase1, spike in phase2, fluctuate then low in phase3
entropy = np.zeros_like(timesteps, dtype=float)
entropy[:15] = np.random.uniform(0.05, 0.2, size=15)  # Phase1: <0.2
entropy[15:25] = np.linspace(0.25, 0.6, 10)           # Phase2: spike to 0.6
entropy[25:] = np.concatenate([
    np.random.uniform(0.3, 0.5, size=5),  # Fluctuate early phase3
    np.random.uniform(0.1, 0.2, size=5)   # Settle low late phase3
])

# Computational depth (layers used: 4=shallow, 32=full depth)
comp_depth = np.zeros_like(timesteps, dtype=int)
comp_depth[:15] = 4  # Phase1: exit at Layer4 (shallow)
comp_depth[15:25] = 32 # Phase2: full depth (Layer32)
comp_depth[25:] = np.concatenate([
    [32, 32, 16, 16, 8],  # Phase3 early: reduce depth gradually
    [4, 4, 4, 4, 4]       # Phase3 late: back to Layer4
])

# Exit decision markers (x=timestep, y=entropy value)
exit_markers_x = [5, 10, 15, 18, 22, 25, 30, 35]
exit_markers_y = [entropy[4], entropy[9], entropy[14], 
                  entropy[17], entropy[21], entropy[24], 
                  entropy[29], entropy[34]]

# --------------------------
# 3. Create 3-Panel Plot
# --------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, 
                         gridspec_kw={"height_ratios": [2, 1.5, 1]})
fig.suptitle("Dynamic Computational Depth During Navigation in Matterport3D", 
             fontsize=12, fontweight="bold", y=0.98)

# --------------------------
# Panel 1: Top - Matterport3D Indoor Scene + Agent Trajectory
# --------------------------
ax1 = axes[0]
ax1.set_ylim(0, 10)
ax1.set_xlim(0, 35)
ax1.set_ylabel("Scene Y-Coordinate", fontsize=9)
ax1.set_title("(Top) Agent Trajectory in Matterport3D Indoor Scene", fontsize=9, pad=10)

# Draw Matterport3D-style room layout with detailed walls and architectural features
# Main hallway with rooms
ax1.add_patch(Rectangle((0, 0), 35, 10, fill=False, linewidth=2, color="black"))  # Outer walls

# Left side: Living room (Phase1 area)
ax1.add_patch(Rectangle((5, 2), 8, 6, fill=False, linewidth=1.5, color="gray"))  # Living room walls
ax1.add_patch(Rectangle((5, 5), 1, 2, fill=True, color="white", linewidth=1.5, edgecolor="gray"))  # Window

# Right side: Dining area (Phase3 area)
ax1.add_patch(Rectangle((22, 2), 8, 6, fill=False, linewidth=1.5, color="gray"))  # Dining area walls
ax1.add_patch(Rectangle((29, 5), 1, 2, fill=True, color="white", linewidth=1.5, edgecolor="gray"))  # Window

# Middle: Obstacle area (Phase2 area) with doorway to kitchen
ax1.add_patch(Rectangle((16, 0), 4, 2, fill=False, linewidth=1.5, color="gray"))  # Kitchen doorway
ax1.add_patch(Rectangle((18, 0), 2, 0.5, fill=True, color="tan", linewidth=1.5, edgecolor="gray"))  # Kitchen threshold

# Detailed Matterport3D-style furniture
# Living room furniture (left side)
ax1.add_patch(Rectangle((6, 6), 4, 2, color="tan", alpha=0.8))  # Sofa
ax1.add_patch(Rectangle((6, 5.5), 1, 0.5, color="saddlebrown", alpha=0.7))  # Coffee table
ax1.add_patch(Rectangle((11, 6), 1.5, 2, color="saddlebrown", alpha=0.7))  # Armchair
ax1.add_patch(Rectangle((8, 4), 2, 1, color="peru", alpha=0.6))  # Side table

# Obstacle area (middle - Phase2)
ax1.add_patch(Rectangle((15, 3), 2, 1.5, color="saddlebrown", alpha=0.8))  # Dining chair 1
ax1.add_patch(Rectangle((18, 5), 3, 1.5, color="peru", alpha=0.8))  # Dining table
ax1.add_patch(Rectangle((21, 3), 2, 1.5, color="saddlebrown", alpha=0.8))  # Dining chair 2
ax1.add_patch(Rectangle((17, 2), 1, 1, color="green", alpha=0.7))  # Potted plant

# Dining area furniture (right side)
ax1.add_patch(Rectangle((23, 6), 2, 2, color="saddlebrown", alpha=0.7))  # Dining chair 3
ax1.add_patch(Rectangle((26, 6), 2, 2, color="saddlebrown", alpha=0.7))  # Dining chair 4
ax1.add_patch(Rectangle((24, 4), 3, 2, color="peru", alpha=0.8))  # Small dining table

# Agent trajectory (dotted line) + agent position markers
trajectory_y = np.concatenate([
    np.full(15, 5),  # Phase1: straight line through hallway
    np.linspace(5, 7, 10),  # Phase2: maneuver around obstacles
    np.full(10, 7)   # Phase3: straight line to destination
])
ax1.plot(timesteps, trajectory_y, color="navy", linestyle=":", linewidth=2, label="Agent Trajectory")

# Enhanced agent position markers with direction
for i, (x, y) in enumerate(zip(timesteps[::5], trajectory_y[::5])):
    ax1.scatter(x, y, color="red", s=30, zorder=5, label="Agent Position" if i == 0 else "")
    ax1.plot([x-0.5, x+0.5], [y-0.2, y+0.2], color="red", linewidth=1.5, zorder=5)
    ax1.plot([x+0.5, x+0.5], [y-0.2, y+0.2], color="red", linewidth=1.5, zorder=5)

# Phase labels with Matterport3D context
ax1.text(7.5, 1.5, "Phase 1\n(Open Hallway)", ha="center", fontsize=8, 
         bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.8))
ax1.text(20, 1.5, "Phase 2\n(Obstacle Avoidance)", ha="center", fontsize=8, 
         bbox=dict(boxstyle="round,pad=0.3", fc="lightcoral", alpha=0.8))
ax1.text(30, 1.5, "Phase 3\n(Recovery)", ha="center", fontsize=8, 
         bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.8))

# Matterport3D scan reference
ax1.text(1, 9, "Matterport3D Scan: MP3D-1234", fontsize=8, color="gray", style="italic")

ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, linestyle=":", alpha=0.3)

# --------------------------
# Panel 2: Middle - Entropy H(p) + Exit Decisions
# --------------------------
ax2 = axes[1]
ax2.plot(timesteps, entropy, color="darkorange", linewidth=2.5, label=r"Entropy $H(\mathbf{p})$")
ax2.scatter(exit_markers_x, exit_markers_y, color="red", s=40, marker="x", 
            label="Exit Decision", zorder=5)
ax2.axhline(y=0.2, color="gray", linestyle="--", linewidth=1, label="Low Entropy Threshold")
ax2.set_ylabel(r"Entropy $H(\mathbf{p})$", fontsize=9)
ax2.set_title("(Middle) Entropy Over Time with Exit Decisions", fontsize=9, pad=10)
ax2.legend(loc="upper right", fontsize=8)
ax2.set_ylim(0, 0.7)
ax2.grid(True, linestyle=":", alpha=0.3)

# --------------------------
# Panel 3: Bottom - Computational Depth
# --------------------------
ax3 = axes[2]
ax3.step(timesteps, comp_depth, where="mid", color="forestgreen", linewidth=2.5, label="Computational Depth")
ax3.axhline(y=4, color="blue", linestyle="--", linewidth=1, label="Layer 4 (Shallow)")
ax3.axhline(y=32, color="purple", linestyle="--", linewidth=1, label="Layer 32 (Full Depth)")
ax3.set_xlabel("Timestep", fontsize=9)
ax3.set_ylabel("Computational Depth (Layer)", fontsize=9)
ax3.set_title("(Bottom) Computational Depth Utilized", fontsize=9, pad=10)
ax3.set_ylim(0, 35)
ax3.set_yticks([4, 8, 16, 32])
ax3.legend(loc="upper right", fontsize=8)
ax3.grid(True, linestyle=":", alpha=0.3)

# --------------------------
# 4. Save as PDF (Publication-Quality)
# --------------------------
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
plt.savefig("decision_paths_visualization_corrected.pdf", 
            dpi=300, bbox_inches="tight", format="pdf")
plt.show()