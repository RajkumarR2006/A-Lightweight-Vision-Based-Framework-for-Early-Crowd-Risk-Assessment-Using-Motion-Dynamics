import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

INPUT = "ucsd_calibrated_validation.csv"
OUTPUT = "temporal_change_results.csv"

df = pd.read_csv(INPUT)
GT = "ground_truth"

# Frozen calibration from UCSD Ped1 training data
D_MIN, D_MAX = 0.00326960, 0.06023295
M_MIN, M_MAX = 0.12297923, 0.46781751

def norm(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)

df["Dn"] = norm(df["density"], D_MIN, D_MAX)
df["Mn"] = norm(df["mii"], M_MIN, M_MAX)

# Calculate temporal changes independently inside each video sequence.
# k=5 is deliberately small and matches the CRI smoothing window.
for k in [1, 5, 10]:
    for col in ["Dn", "Mn", "dci"]:
        df[f"d_{col}_{k}"] = (
            df.groupby("sequence")[col]
              .transform(lambda s: s.diff(k).abs())
              .fillna(0)
        )

# Candidate models
df["current_cpi"] = df["Dn"] * (0.5 * df["Mn"] + 0.5 * df["dci"])

# Pure behavioral-change model
df["change_5"] = (
    df["d_Dn_5"] + df["d_Mn_5"] + df["d_dci_5"]
) / 3

# Motion-change dominant
df["motion_change_5"] = (
    0.2 * df["d_Dn_5"] +
    0.6 * df["d_Mn_5"] +
    0.2 * df["d_dci_5"]
)

# Combine existing risk with behavioral change
df["adaptive_5"] = (
    0.6 * df["current_cpi"] +
    0.4 * df["change_5"]
)

df["adaptive_motion_5"] = (
    0.6 * df["current_cpi"] +
    0.4 * df["motion_change_5"]
)

models = [
    "current_cpi",
    "change_5",
    "motion_change_5",
    "adaptive_5",
    "adaptive_motion_5",
]

def evaluate(series, y):
    mask = np.isfinite(series) & np.isfinite(y)
    x = np.asarray(series)[mask]
    yy = np.asarray(y)[mask]

    if len(np.unique(yy)) < 2:
        return np.nan, np.nan

    return roc_auc_score(yy, x), average_precision_score(yy, x)

print("\n=== TEMPORAL CHANGE EXPERIMENT ===")
print("Frozen calibration; no test-label fitting.")
print()

y = df[GT].astype(int).to_numpy()

print("OVERALL")
print(f"{'Model':22s} {'ROC-AUC':>10s} {'PR-AUC':>10s}")
print("-" * 46)

overall = []
for model in models:
    roc, pr = evaluate(df[model].to_numpy(), y)
    overall.append({
        "scope": "overall",
        "sequence": "ALL",
        "model": model,
        "roc_auc": roc,
        "pr_auc": pr
    })
    print(f"{model:22s} {roc:10.4f} {pr:10.4f}")

print("\nPER SEQUENCE")
per_sequence = []

for seq, g in df.groupby("sequence"):
    yy = g[GT].astype(int).to_numpy()

    print(f"\n{seq}")
    print(f"{'Model':22s} {'ROC-AUC':>10s} {'PR-AUC':>10s}")

    for model in models:
        roc, pr = evaluate(g[model].to_numpy(), yy)
        per_sequence.append({
            "scope": "sequence",
            "sequence": seq,
            "model": model,
            "roc_auc": roc,
            "pr_auc": pr
        })
        roc_s = "N/A" if np.isnan(roc) else f"{roc:.4f}"
        pr_s = "N/A" if np.isnan(pr) else f"{pr:.4f}"
        print(f"{model:22s} {roc_s:>10s} {pr_s:>10s}")

# Specifically inspect Test023
print("\n=== TEST023 ===")
t23 = df[df["sequence"].astype(str).str.contains("Test023", case=False)].copy()
for model in models:
    normal = t23[t23[GT] == 0][model].mean()
    abnormal = t23[t23[GT] == 1][model].mean()
    print(
        f"{model:22s} normal={normal:.4f} "
        f"abnormal={abnormal:.4f}"
    )

# Save compact result table.
results = pd.DataFrame(overall + per_sequence)
results.to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
