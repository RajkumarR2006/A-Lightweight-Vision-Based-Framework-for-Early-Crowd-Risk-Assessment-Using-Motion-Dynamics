from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "UCSD_Anomaly_Dataset.v1p2"
TRAIN = DATASET / "UCSDped1" / "Train"
VALIDATION = ROOT / "ucsd_calibrated_validation.csv"
OUTPUT = ROOT / "normal_model_results.csv"

def compute_features(folder):
    files = sorted(folder.glob("*.tif"))
    bg = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )
    prev = None
    rows = []

    for i, f in enumerate(files):
        frame = cv2.imread(str(f))
        if frame is None:
            continue

        mask = bg.apply(frame)
        density = float(np.count_nonzero(mask > 0) / mask.size)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev is None:
            mii, dci = 0.0, 0.0
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, angle = cv2.cartToPolar(
                flow[..., 0], flow[..., 1]
            )

            mii = float(np.std(magnitude))

            valid = magnitude > 0.5
            if np.any(valid):
                a = angle[valid]
                w = magnitude[valid]
                mean_cos = np.average(np.cos(a), weights=w)
                mean_sin = np.average(np.sin(a), weights=w)
                r = np.sqrt(mean_cos**2 + mean_sin**2)
                dci = float(np.clip(1.0 - r, 0, 1))
            else:
                dci = 0.0

        rows.append([i + 1, density, mii, dci])
        prev = gray

    return pd.DataFrame(rows, columns=["frame", "density", "mii", "dci"])

print("\n=== LEARNING NORMAL BEHAVIOR FROM UCSD TRAIN ===\n")

train_parts = []

for folder in sorted(TRAIN.iterdir()):
    if not folder.is_dir():
        continue

    try:
        x = compute_features(folder)
        x["sequence"] = folder.name
        train_parts.append(x)
        print("Processed:", folder.name, "| frames:", len(x))
    except Exception as e:
        print("Skipped:", folder.name, "|", e)

if not train_parts:
    raise RuntimeError("No UCSD training sequences were processed.")

train = pd.concat(train_parts, ignore_index=True)

features = ["density", "mii", "dci"]

# Robust normal statistics using median + MAD.
stats = {}
for feature in features:
    median = float(train[feature].median())
    mad = float(np.median(np.abs(train[feature] - median)))
    scale = 1.4826 * mad

    if scale < 1e-6:
        scale = float(train[feature].std())

    stats[feature] = (median, max(scale, 1e-6))

print("\nNORMAL DISTRIBUTION")
for feature in features:
    median, scale = stats[feature]
    print(f"{feature:8s}: median={median:.6f}, scale={scale:.6f}")

# Load the already-generated test feature CSV.
df = pd.read_csv(VALIDATION)

# Do NOT use ground truth for fitting.
anomaly_scores = []

for _, row in df.iterrows():
    z_values = []

    for feature in features:
        median, scale = stats[feature]
        z = abs(float(row[feature]) - median) / scale
        z_values.append(z)

    z_mean = np.mean(z_values)

    # Compress robust z-score into [0,1].
    score = 1.0 - np.exp(-z_mean / 2.0)
    anomaly_scores.append(float(np.clip(score, 0, 1)))

df["normal_anomaly_score"] = anomaly_scores

# Combine learned anomaly score with the existing CPI.
df["cpi_plus_anomaly"] = (
    0.7 * df["cpi"] +
    0.3 * df["normal_anomaly_score"]
)

def evaluate(x, y):
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return (
        roc_auc_score(y, x),
        average_precision_score(y, x)
    )

models = ["cpi", "normal_anomaly_score", "cpi_plus_anomaly"]
y = df["ground_truth"].astype(int)

print("\n=== OVERALL RESULTS ===")
print(f"{'MODEL':24s} {'ROC-AUC':>10s} {'PR-AUC':>10s}")
print("-" * 48)

for model in models:
    roc, pr = evaluate(df[model], y)
    print(f"{model:24s} {roc:10.4f} {pr:10.4f}")

print("\n=== PER-SEQUENCE RESULTS ===")

for seq, g in df.groupby("sequence"):
    print(f"\n{seq}")

    for model in models:
        roc, pr = evaluate(
            g[model].to_numpy(),
            g["ground_truth"].astype(int).to_numpy()
        )

        if np.isnan(roc):
            print(f"{model:24s} N/A")
        else:
            print(
                f"{model:24s} "
                f"ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}"
            )

print("\n=== TEST023 ===")

t23 = df[df["sequence"] == "Test023"]

for model in models:
    normal_mean = t23[t23["ground_truth"] == 0][model].mean()
    abnormal_mean = t23[t23["ground_truth"] == 1][model].mean()

    print(
        f"{model:24s} "
        f"normal={normal_mean:.4f}  "
        f"abnormal={abnormal_mean:.4f}"
    )

df.to_csv(OUTPUT, index=False)

print(f"\nSaved: {OUTPUT}")
print("Done.")
