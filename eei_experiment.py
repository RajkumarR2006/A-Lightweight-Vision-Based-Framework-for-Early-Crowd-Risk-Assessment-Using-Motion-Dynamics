
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

INPUT_CANDIDATES = [
    OUTPUT_DIR / "normal_model_results.csv",
    PROJECT_ROOT / "normal_model_results.csv",
]

GT_START = {
    "Test003": 91,
    "Test004": 31,
    "Test014": 1,
    "Test018": 54,
    "Test019": 64,
    "Test021": 31,
    "Test022": 16,
    "Test023": 8,
    "Test024": 50,
    "Test032": 1,
}


def load_input():
    for path in INPUT_CANDIDATES:
        if path.exists():
            print(f"Using: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(
        "normal_model_results.csv not found. Put it in the outputs folder "
        "or beside this script."
    )


def first_sustained_warning(frames, flags, persistence=3):
    run = 0
    for frame, flag in zip(frames, flags):
        run = run + 1 if flag else 0
        if run >= persistence:
            return int(frame)
    return None


def evaluate(df, window, quantile):
    rows = []

    for seq, g in df.groupby("sequence", sort=True):
        g = g.sort_values("frame").copy()
        gt = GT_START.get(seq)
        if gt is None:
            continue

        # CRI escalation over a temporal horizon.
        g["delta"] = g["cri"] - g["cri"].shift(window)

        # Local baseline: recent CRI median.
        g["local_baseline"] = g["cri"].rolling(
            window, min_periods=window
        ).median()

        g["local_rise"] = g["cri"] - g["local_baseline"]

        # A simple interpretable Early Escalation Index:
        # positive local rise + positive longer-term change.
        # We do NOT call this congestion by itself.
        scale = g["cri"].rolling(window, min_periods=window).std()
        scale = scale.replace(0, np.nan).fillna(
            g["cri"].std() if g["cri"].std() > 0 else 1e-6
        )

        g["eei"] = (
            0.5 * np.clip(g["local_rise"] / scale, -5, 5)
            + 0.5 * np.clip(g["delta"] / scale, -5, 5)
        )

        # Exploratory threshold: calibrated from the sequence's NORMAL
        # portion only. This is deliberately labelled exploratory and
        # must not be used as the final reported test metric.
        pre = g.loc[g["frame"] < gt, "eei"].dropna()

        if len(pre) < 5:
            rows.append({
                "sequence": seq,
                "window": window,
                "quantile": quantile,
                "gt_start": gt,
                "threshold": np.nan,
                "first_warning": np.nan,
                "lead_frames": np.nan,
                "false_alarm_frames": np.nan,
                "max_eei": float(g["eei"].max()),
                "status": "insufficient pre-GT frames",
            })
            continue

        threshold = float(pre.quantile(quantile))

        flags = (g["eei"] >= threshold).fillna(False).to_numpy()
        frames = g["frame"].to_numpy()

        first = first_sustained_warning(frames, flags, persistence=3)

        pre_flags = g.loc[g["frame"] < gt, "eei"] >= threshold
        false_alarm = int(pre_flags.sum())

        lead = gt - first if first is not None else np.nan

        # Only a warning strictly before GT counts as an early warning.
        if first is None:
            status = "NO WARNING"
        elif first < gt:
            status = "EARLY"
        else:
            status = "LATE"

        rows.append({
            "sequence": seq,
            "window": window,
            "quantile": quantile,
            "gt_start": gt,
            "threshold": threshold,
            "first_warning": first,
            "lead_frames": lead,
            "false_alarm_frames": false_alarm,
            "max_eei": float(g["eei"].max()),
            "status": status,
        })

    return pd.DataFrame(rows)


def main():
    print("=" * 72)
    print("CROWDSENSE - EARLY ESCALATION INDEX (EEI) EXPERIMENT")
    print("=" * 72)
    print()
    print("Purpose:")
    print("  Test whether RISK ESCALATION contains an earlier signal than")
    print("  an absolute CRI threshold.")
    print()
    print("Important:")
    print("  This is an exploratory diagnostic experiment.")
    print("  UCSD ground-truth events are anomaly labels, not guaranteed")
    print("  congestion-onset labels.")
    print()

    df = load_input()

    required = {"sequence", "frame", "cri"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    all_results = []

    for window in [5, 10, 15, 20]:
        for quantile in [0.90, 0.95, 0.99]:
            print(f"Testing window={window}, normal quantile={quantile:.2f}")
            all_results.append(evaluate(df, window, quantile))

    results = pd.concat(all_results, ignore_index=True)

    output = OUTPUT_DIR / "eei_experiment_results.csv"
    results.to_csv(output, index=False)

    # Compact comparison.
    summary_rows = []
    for (window, quantile), g in results.groupby(["window", "quantile"]):
        valid = g[g["status"] != "insufficient pre-GT frames"]
        early = valid[valid["status"] == "EARLY"]

        summary_rows.append({
            "window": window,
            "normal_quantile": quantile,
            "sequences_evaluated": len(valid),
            "sequences_with_early_warning": len(early),
            "mean_positive_lead_frames": (
                float(early["lead_frames"].mean())
                if len(early) else 0.0
            ),
            "mean_false_alarm_frames_before_gt": (
                float(valid["false_alarm_frames"].mean())
                if len(valid) else np.nan
            ),
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        ["sequences_with_early_warning", "mean_positive_lead_frames"],
        ascending=[False, False],
    )

    summary_output = OUTPUT_DIR / "eei_experiment_summary.csv"
    summary.to_csv(summary_output, index=False)

    print()
    print("=" * 72)
    print("EEI SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False))

    print()
    print("Best exploratory configuration:")
    best = summary.iloc[0]
    print(
        f"  window={int(best['window'])}, "
        f"quantile={best['normal_quantile']:.2f}, "
        f"early warnings={int(best['sequences_with_early_warning'])}/"
        f"{int(best['sequences_evaluated'])}"
    )

    print()
    print("Saved:")
    print(f"  {output}")
    print(f"  {summary_output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
