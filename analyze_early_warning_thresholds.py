from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(r"E:\PROJECTS\OPEN_CV\datasets\UCSD_Anomaly_Dataset.v1p2")
TRAIN_ROOT = DATASET_ROOT / "UCSDped1" / "Train"
TEST_ROOT = DATASET_ROOT / "UCSDped1" / "Test"
OUTPUT = PROJECT_ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

DENSITY_MIN, DENSITY_MAX = 0.00326960, 0.06023295
MII_MIN, MII_MAX = 0.12297923, 0.46781751
WINDOW = 5
FARNEBACK = (0.5, 3, 15, 3, 5, 1.2, 0)

ABNORMAL_SEQUENCES = {
    "Test003": 91, "Test004": 31, "Test014": 1,
    "Test018": 54, "Test019": 64, "Test021": 31,
    "Test022": 16, "Test023": 8, "Test024": 50,
    "Test032": 1,
}


def extract(seq):
    files = sorted(Path(seq).glob("*.tif"))
    sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=False
    )
    kernel = np.ones((3, 3), np.uint8)
    prev = None
    out = []

    for n, f in enumerate(files, 1):
        frame = cv2.imread(str(f))
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = sub.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        d = np.count_nonzero(mask > 0) / mask.size

        if prev is None:
            out.append([n, d, 0.0, 0.0])
            prev = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, *FARNEBACK)
        mag, ang = cv2.cartToPolar(
            flow[..., 0], flow[..., 1], angleInDegrees=True
        )

        mi = float(np.std(mag.astype(np.float32)))
        threshold = max(float(np.percentile(mag, 50)), 1e-6)
        valid = mag > threshold

        if np.any(valid):
            theta = np.deg2rad(ang[valid])
            c = float(np.mean(np.cos(theta)))
            s = float(np.mean(np.sin(theta)))
            dc = float(np.clip(1 - math.sqrt(c*c + s*s), 0, 1))
        else:
            dc = 0.0

        out.append([n, d, mi, dc])
        prev = gray

    return pd.DataFrame(out, columns=["frame", "density", "mii", "dci"])


def learn_normal():
    parts = []
    seqs = sorted(p for p in TRAIN_ROOT.iterdir() if p.is_dir())

    for i, seq in enumerate(seqs, 1):
        print(f"TRAIN {i:02d}/{len(seqs)}: {seq.name}")
        parts.append(extract(seq))

    train = pd.concat(parts, ignore_index=True)
    stats = {}

    for f in ["density", "mii", "dci"]:
        x = train[f].astype(float)
        med = float(x.median())
        mad = float(np.median(np.abs(x - med)))
        stats[f] = (med, max(1.4826 * mad, 1e-6))

    return stats


def score(df, stats):
    dn = ((df.density - DENSITY_MIN) /
          (DENSITY_MAX - DENSITY_MIN)).clip(0, 1)
    mn = ((df.mii - MII_MIN) /
          (MII_MAX - MII_MIN)).clip(0, 1)
    dc = df.dci.clip(0, 1)

    df["cpi"] = dn * (0.5 * mn + 0.5 * dc)

    a = []
    for _, r in df.iterrows():
        z = []
        for f in ["density", "mii", "dci"]:
            med, scale = stats[f]
            z.append(abs(float(r[f]) - med) / scale)
        a.append(1 - np.exp(-float(np.mean(z)) / 2))

    df["anomaly"] = np.clip(a, 0, 1)
    df["risk"] = np.clip(0.7 * df.cpi + 0.3 * df.anomaly, 0, 1)
    df["cri"] = df.risk.rolling(WINDOW, min_periods=1).mean()
    return df


def first_sustained_warning(cri, threshold, consecutive=3):
    flags = (cri >= threshold).astype(int)
    rolling = flags.rolling(consecutive).sum().fillna(0)
    hits = np.where(rolling.values == consecutive)[0]
    if len(hits) == 0:
        return None
    return int(cri.index[hits[0]])


def main():
    print("=" * 70)
    print("CROWDSENSE - EARLY WARNING THRESHOLD ANALYSIS")
    print("=" * 70)

    stats = learn_normal()

    # Build TRAIN-only CRI distribution.
    print("\nBuilding TRAIN-only CRI distribution...")
    train_cri = []
    for seq in sorted(p for p in TRAIN_ROOT.iterdir() if p.is_dir()):
        train_cri.append(score(extract(seq), stats).cri)

    train_cri = pd.concat(train_cri, ignore_index=True)

    # Candidate thresholds are based ONLY on normal TRAIN CRI.
    quantiles = [0.95, 0.97, 0.98, 0.985, 0.99, 0.995]
    thresholds = {
        q: float(train_cri.quantile(q))
        for q in quantiles
    }

    print("\nTRAIN-only candidate thresholds:")
    for q, t in thresholds.items():
        print(f"  {q:.3f} -> {t:.6f}")

    rows = []

    for seq_name, gt_start in ABNORMAL_SEQUENCES.items():
        print(f"\nTEST: {seq_name} (GT starts frame {gt_start})")

        df = score(
            extract(TEST_ROOT / seq_name),
            stats
        )

        y = (df.frame >= gt_start).astype(int)

        for q, threshold in thresholds.items():

            # First sustained threshold crossing.
            idx = first_sustained_warning(
                df.cri,
                threshold,
                consecutive=3
            )

            warning_frame = (
                int(df.loc[idx, "frame"])
                if idx is not None
                else None
            )

            lead = (
                gt_start - warning_frame
                if warning_frame is not None
                else np.nan
            )

            pre = df[df.frame < gt_start]
            false_alarm = int(
                np.count_nonzero(pre.cri >= threshold)
            )

            post = df[df.frame >= gt_start]
            detection = float(
                np.mean(post.cri >= threshold)
            ) if len(post) else np.nan

            rows.append([
                seq_name, q, threshold, gt_start,
                warning_frame, lead, false_alarm,
                detection, float(df.cri.max())
            ])

    result = pd.DataFrame(rows, columns=[
        "sequence", "normal_quantile", "threshold",
        "gt_start", "first_sustained_warning",
        "lead_frames", "false_alarm_frames",
        "post_gt_detection_rate", "max_cri"
    ])

    result.to_csv(
        OUTPUT / "early_warning_threshold_comparison.csv",
        index=False
    )

    print("\n" + "=" * 70)
    print("AGGREGATED EARLY-WARNING RESULTS")
    print("=" * 70)

    summary = []

    for q in quantiles:
        r = result[result.normal_quantile == q]

        valid_leads = r.lead_frames.dropna()

        # Positive lead means warning before ground truth.
        early = valid_leads[valid_leads > 0]

        summary.append([
            q,
            float(r.threshold.iloc[0]),
            int(r.first_sustained_warning.notna().sum()),
            int((r.lead_frames > 0).sum()),
            float(early.mean()) if len(early) else np.nan,
            float(r.false_alarm_frames.mean()),
            float(r.post_gt_detection_rate.mean()),
            float(r.max_cri.mean())
        ])

    summary_df = pd.DataFrame(summary, columns=[
        "normal_quantile", "threshold",
        "sequences_with_warning",
        "sequences_with_early_warning",
        "mean_positive_lead_frames",
        "mean_false_alarm_frames_before_gt",
        "mean_post_gt_detection_rate",
        "mean_max_cri"
    ])

    summary_df.to_csv(
        OUTPUT / "early_warning_threshold_summary.csv",
        index=False
    )

    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nFiles saved:")
    print(OUTPUT / "early_warning_threshold_comparison.csv")
    print(OUTPUT / "early_warning_threshold_summary.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
