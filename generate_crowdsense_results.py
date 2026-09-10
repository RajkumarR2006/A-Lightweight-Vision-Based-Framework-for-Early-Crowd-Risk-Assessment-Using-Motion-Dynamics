from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

INPUT = "normal_model_results.csv"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(INPUT)

# Use the final experimentally selected score.
y = df["ground_truth"].astype(int)
risk = df["cpi_plus_anomaly"]

# 1. ROC curve
fpr, tpr, _ = roc_curve(y, risk)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"Combined Risk (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CrowdSense ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curve.png", dpi=200)
plt.close()

# 2. Precision-Recall curve
precision, recall, _ = precision_recall_curve(y, risk)
pr_auc = auc(recall, precision)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, label=f"Combined Risk (AP={pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("CrowdSense Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "pr_curve.png", dpi=200)
plt.close()

# 3. Example risk timeline: Test003
seq = "Test003"
g = df[df["sequence"] == seq].sort_values("frame")

plt.figure(figsize=(10, 5))
plt.plot(g["frame"], g["cpi"], label="CPI")
plt.plot(g["frame"], g["cpi_plus_anomaly"], label="Combined Risk")
plt.plot(g["frame"], g["ground_truth"], label="Ground Truth")
plt.xlabel("Frame")
plt.ylabel("Score")
plt.title(f"CrowdSense Risk Timeline - {seq}")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "test003_risk_timeline.png", dpi=200)
plt.close()

# 4. Per-sequence summary
summary = (
    df.groupby("sequence")
      .agg(
          frames=("frame", "count"),
          mean_cpi=("cpi", "mean"),
          mean_risk=("cpi_plus_anomaly", "mean"),
          abnormal_frames=("ground_truth", "sum")
      )
      .reset_index()
)
summary.to_csv(OUT_DIR / "risk_summary.csv", index=False)

print("=== VISUALIZATION COMPLETE ===")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC : {pr_auc:.4f}")
print()
print("Created:")
print("outputs/roc_curve.png")
print("outputs/pr_curve.png")
print("outputs/test003_risk_timeline.png")
print("outputs/risk_summary.csv")
