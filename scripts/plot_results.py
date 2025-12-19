"""
Scatter plot of evaluation results.
Plots Taiwan Traffic mAP50-95 (x-axis) vs COCO Subset mAP50-95 (y-axis).
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_path = Path("results/all_results.json")
with open(results_path, "r") as f:
    all_results = json.load(f)

# Extract data for plotting
taiwan_map = []
coco_subset_map = []
labels = []

for entry in all_results:
    model_name = Path(entry["model_path"]).parent.parent.name  # Get experiment name
    results = entry["results"]

    taiwan_map.append(results["Taiwan mAP50-95"])
    coco_subset_map.append(results["COCO Subset mAP50-95"])
    labels.append(model_name)

# Create scatter plot
plt.figure(figsize=(12, 8))

# Plot points
scatter = plt.scatter(
    taiwan_map,
    coco_subset_map,
    s=100,
    alpha=0.7,
    c="steelblue",
    edgecolors="darkblue",
    linewidths=1.5,
)

# Add labels for each point
for i, label in enumerate(labels):
    plt.annotate(
        label,
        (taiwan_map[i], coco_subset_map[i]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=11,
        alpha=0.8,
    )

# Styling
plt.xlabel("Taiwan Traffic mAP50-95", fontsize=12, fontweight="bold")
plt.ylabel("COCO Subset mAP50-95", fontsize=12, fontweight="bold")
plt.title(
    "Model Performance: Taiwan Traffic vs COCO Subset", fontsize=14, fontweight="bold"
)

# Add grid
plt.grid(True, alpha=0.3, linestyle="--")

# Tight layout
plt.tight_layout()

# Save figure
output_path = Path("results/scatter_taiwan_vs_coco_without_yolo11m.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {output_path}")

# Show plot
plt.show()

# Print summary table
print("\n" + "=" * 60)
print("Summary Table")
print("=" * 60)
print(f"{'Model':<50} {'Taiwan':<10} {'COCO':<10}")
print("-" * 60)
for i, label in enumerate(labels):
    print(f"{label:<50} {taiwan_map[i]:.4f}     {coco_subset_map[i]:.4f}")
