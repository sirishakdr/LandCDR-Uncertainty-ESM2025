import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
INPUT_ESM_NAME = "cumulative_carbon_density_afforestation_global_ESM_ensemble_paper.csv"
INPUT_IAM_NAME = "cumulative_carbon_density_afforestation_global_IAM_paper.csv"
OUTPUT_XLSX_NAME = "scaling_factors_afforestation_global_paper.xlsx"

# ---------------- Path resolution (portable, no local hardcoded path) ----------------
def resolve_data_dir(required_files):
    candidates = []

    # Optional override:
    # export PAPER_FILES_DIR=/path/to/paper_files
    env_dir = os.getenv("PAPER_FILES_DIR")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        candidates.append(p)
        candidates.append(p / "paper_files")

    # Notebook / terminal current directory
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.append(cwd / "paper_files")

    # Script execution directory
    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
        candidates.append(script_dir / "paper_files")
        candidates.append(script_dir.parent / "paper_files")

    # De-duplicate preserving order
    unique = []
    seen = set()
    for c in candidates:
        k = str(c)
        if k not in seen:
            seen.add(k)
            unique.append(c)

    for d in unique:
        if all((d / f).exists() for f in required_files):
            return d

    checked = "\n".join(str(d) for d in unique)
    raise FileNotFoundError(
        "Could not locate required input files.\n"
        "Set PAPER_FILES_DIR or run from a folder containing paper_files.\n"
        "Expected files:\n"
        + "\n".join(f"  - {f}" for f in required_files)
        + "\nChecked locations:\n"
        + checked
    )


DATA_DIR = resolve_data_dir([INPUT_ESM_NAME, INPUT_IAM_NAME])
ESM_CSV = DATA_DIR / INPUT_ESM_NAME
IAM_CSV = DATA_DIR / INPUT_IAM_NAME
OUTPUT_XLSX = DATA_DIR / OUTPUT_XLSX_NAME

# ---------------- Helpers ----------------
def find_col(df, candidates, required=True, label="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find {label}. Tried: {candidates}. Available: {list(df.columns)}")
    return None

def normalize_ssp(v):
    raw = str(v).strip()
    low = raw.lower().replace("_", "").replace("-", "").replace(" ", "")

    direct_map = {
        "ssp126": "SSP1-26",
        "ssp245": "SSP2-45",
        "ssp370": "SSP3-70",
        "ssp585": "SSP5-85",
    }
    if low in direct_map:
        return direct_map[low]

    m = re.search(r"ssp([1-5])(\d{2})", low)
    if m:
        return f"SSP{m.group(1)}-{m.group(2)}"

    return raw

# ---------------- Load ----------------
esm_df = pd.read_csv(ESM_CSV)
iam_df = pd.read_csv(IAM_CSV)

print("ESM columns:", esm_df.columns.tolist())
print("IAM columns:", iam_df.columns.tolist())

# Resolve columns robustly
esm_ssp_col = find_col(esm_df, ["SSP", "Scenario", "scenario"], True, "ESM SSP column")
esm_carbon_col = find_col(
    esm_df,
    ["carbon_density_ESM", "CumulativeCarbonDensity_tCO2_per_ha", "Carbon_Density_tCO2_per_ha"],
    True,
    "ESM carbon density column",
)

iam_ssp_col = find_col(iam_df, ["Scenario", "scenario", "SSP", "SSP_std"], True, "IAM scenario column")
iam_model_col = find_col(iam_df, ["Model", "model", "IAM_model"], True, "IAM model column")
iam_carbon_col = find_col(
    iam_df,
    ["Carbon_Density_IAM_TCO2_ha", "IAM_CarbonDensity", "Cumulative_Carbon_Density_2020_2100_tCO2_ha"],
    True,
    "IAM carbon density column",
)

# Standardize SSP labels
esm_df["SSP_std"] = esm_df[esm_ssp_col].apply(normalize_ssp)
iam_df["SSP_std"] = iam_df[iam_ssp_col].apply(normalize_ssp)

# Standardize and numeric conversion
esm_df["ESM_CarbonDensity"] = pd.to_numeric(esm_df[esm_carbon_col], errors="coerce")
iam_df = iam_df.rename(columns={iam_model_col: "IAM_model", iam_carbon_col: "IAM_CarbonDensity"})
iam_df["IAM_CarbonDensity"] = pd.to_numeric(iam_df["IAM_CarbonDensity"], errors="coerce")

# Merge on standardized SSP
merged = esm_df.merge(
    iam_df[["IAM_model", "SSP_std", "IAM_CarbonDensity"]],
    on="SSP_std",
    how="inner",
)

if merged.empty:
    raise ValueError("Merge returned zero rows. Check SSP labels and input files.")

# Scaling factor
merged["scaling_factor"] = np.where(
    merged["IAM_CarbonDensity"].isna() | (merged["IAM_CarbonDensity"] == 0),
    np.nan,
    merged["ESM_CarbonDensity"] / merged["IAM_CarbonDensity"],
)

# Save to Excel in DATA_DIR
merged.to_excel(OUTPUT_XLSX, index=False)

print(f"Saved scaling factors to {OUTPUT_XLSX_NAME}")
print(f"Rows: {len(merged)}")
print("Rows by SSP:")
print(merged["SSP_std"].value_counts(dropna=False))