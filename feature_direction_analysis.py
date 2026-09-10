import pandas as pd
import numpy as np

INPUT = "ucsd_calibrated_validation.csv"
OUTPUT = "feature_direction_analysis.csv"

df = pd.read_csv(INPUT)

GT_COL = "ground_truth"
FEATURES = ["density", "mii", "dci", "cpi", "cri"]

rows = []

for seq, g in df.groupby("sequence"):
    normal = g[g[GT_COL] == 0]
    abnormal = g[g[GT_COL] == 1]

    row = {
        "sequence": seq,
        "normal_frames": len(normal),
        "abnormal_frames": len(abnormal),
    }

    for feature in FEATURES:
        n = normal[feature].mean() if len(normal) else np.nan
        a = abnormal[feature].mean() if len(abnormal) else np.nan

        row[f"{feature}_normal_mean"] = n
        row[f"{feature}_abnormal_mean"] = a

        if pd.isna(n) or pd.isna(a):
            direction = "N/A"
        elif a > n:
            direction = "HIGHER"
        elif a < n:
            direction = "LOWER"
        else:
            direction = "SAME"

        row[f"{feature}_direction"] = direction

        if not pd.isna(n) and n != 0:
            row[f"{feature}_relative_change_pct"] = ((a - n) / abs(n)) * 100
        else:
            row[f"{feature}_relative_change_pct"] = np.nan

    rows.append(row)

result = pd.DataFrame(rows)
result.to_csv(OUTPUT, index=False)

print("\n=== FEATURE DIRECTION ANALYSIS ===\n")
print("For each sequence:")
print("HIGHER = abnormal frames have higher feature values")
print("LOWER  = abnormal frames have lower feature values")
print()

for feature in FEATURES:
    counts = result[f"{feature}_direction"].value_counts()
    print(
        f"{feature.upper():8s} | "
        f"HIGHER: {counts.get('HIGHER', 0):2d} | "
        f"LOWER: {counts.get('LOWER', 0):2d} | "
        f"SAME/N/A: {counts.get('SAME', 0) + counts.get('N/A', 0):2d}"
    )

print("\n=== PER-SEQUENCE RESULTS ===\n")

for _, row in result.iterrows():
    print(f"{row['sequence']}:")
    for feature in FEATURES:
        direction = row[f"{feature}_direction"]
        change = row[f"{feature}_relative_change_pct"]
        print(f"  {feature.upper():8s}: {direction:6s} ({change:+.2f}%)")
    print()

print(f"Saved: {OUTPUT}")
