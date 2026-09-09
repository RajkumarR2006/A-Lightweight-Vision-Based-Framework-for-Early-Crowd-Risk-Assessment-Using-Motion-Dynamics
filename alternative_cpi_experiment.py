import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "ucsd_calibrated_validation.csv"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "cpi_ablation_results.csv"
)

DENSITY_MIN = 0.00326960
DENSITY_MAX = 0.06023295

MII_MIN = 0.12297923
MII_MAX = 0.46781751


def normalize(value, minimum, maximum):
    if maximum <= minimum:
        raise ValueError("Maximum must be greater than minimum.")

    result = (value - minimum) / (maximum - minimum)

    return np.clip(result, 0.0, 1.0)


if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "sequence",
    "frame",
    "mii",
    "density",
    "dci",
    "ground_truth"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Missing column: {column}"
        )


# ============================================================
# NORMALIZE FEATURES USING FROZEN CALIBRATION
# ============================================================

df["density_n"] = normalize(
    df["density"].values,
    DENSITY_MIN,
    DENSITY_MAX
)

df["mii_n"] = normalize(
    df["mii"].values,
    MII_MIN,
    MII_MAX
)

df["dci_n"] = np.clip(
    df["dci"].values,
    0.0,
    1.0
)


# ============================================================
# CPI FORMULATIONS
# ============================================================

# A. Density only
df["density_only"] = df["density_n"]


# B. MII only
df["mii_only"] = df["mii_n"]


# C. DCI only
df["dci_only"] = df["dci_n"]


# D. MII + DCI
df["mii_dci"] = (
    0.5 * df["mii_n"]
    + 0.5 * df["dci_n"]
)


# E. Density + MII
df["density_mii"] = (
    0.5 * df["density_n"]
    + 0.5 * df["mii_n"]
)


# F. Density + DCI
df["density_dci"] = (
    0.5 * df["density_n"]
    + 0.5 * df["dci_n"]
)


# G. Current multiplicative CPI
df["current_cpi"] = (
    df["density_n"]
    * (
        0.5 * df["mii_n"]
        + 0.5 * df["dci_n"]
    )
)


# H. Additive full CPI
df["additive_cpi"] = (
    (1.0 / 3.0) * df["density_n"]
    + (1.0 / 3.0) * df["mii_n"]
    + (1.0 / 3.0) * df["dci_n"]
)


# ============================================================
# CONFIGURATIONS
# ============================================================

configurations = {
    "Density Only": "density_only",
    "MII Only": "mii_only",
    "DCI Only": "dci_only",
    "MII + DCI": "mii_dci",
    "Density + MII": "density_mii",
    "Density + DCI": "density_dci",
    "Current CPI": "current_cpi",
    "Additive CPI": "additive_cpi"
}


# ============================================================
# OVERALL RESULTS
# ============================================================

ground_truth = df["ground_truth"].values

results = []

for name, column in configurations.items():

    score = df[column].values

    roc_auc = roc_auc_score(
        ground_truth,
        score
    )

    pr_auc = average_precision_score(
        ground_truth,
        score
    )

    results.append([
        "Overall",
        name,
        roc_auc,
        pr_auc,
        len(df)
    ])


# ============================================================
# PER-SEQUENCE RESULTS
# ============================================================

for sequence in sorted(
    df["sequence"].unique()
):

    sequence_df = df[
        df["sequence"] == sequence
    ]

    y_true = sequence_df[
        "ground_truth"
    ].values

    for name, column in configurations.items():

        score = sequence_df[
            column
        ].values

        unique_labels = np.unique(
            y_true
        )

        if len(unique_labels) < 2:

            roc_auc = np.nan

        else:

            roc_auc = roc_auc_score(
                y_true,
                score
            )

        pr_auc = average_precision_score(
            y_true,
            score
        )

        results.append([
            sequence,
            name,
            roc_auc,
            pr_auc,
            len(sequence_df)
        ])


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "sequence",
        "configuration",
        "roc_auc",
        "pr_auc",
        "frames"
    ]
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

overall = results_df[
    results_df["sequence"] == "Overall"
].copy()

overall = overall.sort_values(
    "roc_auc",
    ascending=False
)

print()
print("========================================")
print("CROWD SENSE")
print("ALTERNATIVE CPI EXPERIMENT")
print("========================================")

print()
print(
    f"{'Configuration':<22}"
    f"{'ROC-AUC':>10}"
    f"{'PR-AUC':>10}"
)

print("------------------------------------------")

for _, row in overall.iterrows():

    print(
        f"{row['configuration']:<22}"
        f"{row['roc_auc']:>10.4f}"
        f"{row['pr_auc']:>10.4f}"
    )


# ============================================================
# PER-SEQUENCE CURRENT VS ADDITIVE
# ============================================================

print()
print("========================================")
print("CURRENT CPI VS ADDITIVE CPI")
print("========================================")

current = results_df[
    results_df["configuration"] == "Current CPI"
].set_index("sequence")

additive = results_df[
    results_df["configuration"] == "Additive CPI"
].set_index("sequence")

for sequence in sorted(
    df["sequence"].unique()
):

    current_auc = current.loc[
        sequence,
        "roc_auc"
    ]

    additive_auc = additive.loc[
        sequence,
        "roc_auc"
    ]

    print(
        f"{sequence}: "
        f"Current={current_auc:.4f} | "
        f"Additive={additive_auc:.4f}"
    )


print()
print("========================================")
print("EXPERIMENT COMPLETE")
print("========================================")
print("Results saved to:")
print(OUTPUT_FILE)
print("========================================")
