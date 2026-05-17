import os
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------------------------------
# File names
# -------------------------------------------------
INPUT_ESM = "cumulative_carbon_density_biomass_global_ESM_ensemble_paper.xlsx"
INPUT_IAM = "cumulative_carbon_density_biomass_global_IAM_paper.xlsx"
OUTPUT_FILE = "scaling_factors_global_bioenergy_carbon_paper.xlsx"


# -------------------------------------------------
# Find data directory robustly (no hardcoded local path)
# -------------------------------------------------
def find_data_dir(required_files) -> Path:
    candidates = []

    # Optional override for collaborators/CI:
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

    # Script-based execution
    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
        candidates.append(script_dir / "paper_files")
        candidates.append(script_dir.parent / "paper_files")

    # De-duplicate while preserving order
    unique_candidates = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for d in unique_candidates:
        if all((d / f).exists() for f in required_files):
            return d

    raise FileNotFoundError(
        "Could not find required input files. "
        "Place inputs in a paper_files folder or set PAPER_FILES_DIR."
    )


DATA_DIR = find_data_dir([INPUT_ESM, INPUT_IAM])
ESM_XLSX = DATA_DIR / INPUT_ESM
IAM_XLSX = DATA_DIR / INPUT_IAM
OUTPUT_XLSX = DATA_DIR / OUTPUT_FILE


# -------------------------------------------------
# Load
# -------------------------------------------------
esm_df = pd.read_excel(ESM_XLSX)
iam_df = pd.read_excel(IAM_XLSX)

# Clean column names
esm_df.columns = [str(c).strip() for c in esm_df.columns]
iam_df.columns = [str(c).strip() for c in iam_df.columns]

# Validate required columns
required_esm_cols = {"SSP", "CumulativeCarbonDensity_tCO2_per_ha"}
required_iam_cols = {"model", "scenario", "Cumulative_Carbon_Density_2020_2100_tCO2_ha"}

missing_esm = required_esm_cols - set(esm_df.columns)
missing_iam = required_iam_cols - set(iam_df.columns)

if missing_esm:
    raise ValueError(f"ESM input missing columns: {sorted(missing_esm)}")
if missing_iam:
    raise ValueError(f"IAM input missing columns: {sorted(missing_iam)}")


# -------------------------------------------------
# Standardize SSP labels
# -------------------------------------------------
def standardize_ssp(val: str) -> str:
    s = str(val).strip().lower().replace("-", "").replace("_", "")
    ssp_map = {
        "ssp126": "SSP1-26",
        "ssp245": "SSP2-45",
        "ssp370": "SSP3-70",
        "ssp585": "SSP5-85",
    }
    return ssp_map.get(s, str(val).strip())


esm_df["SSP_std"] = esm_df["SSP"].apply(standardize_ssp)
iam_df["SSP_std"] = iam_df["scenario"].astype(str).str.strip()

# Rename IAM columns for merge
iam_df = iam_df.rename(
    columns={
        "Cumulative_Carbon_Density_2020_2100_tCO2_ha": "IAM_CarbonDensity",
        "model": "IAM_model",
    }
)

# Ensure numeric
esm_df["CumulativeCarbonDensity_tCO2_per_ha"] = pd.to_numeric(
    esm_df["CumulativeCarbonDensity_tCO2_per_ha"], errors="coerce"
)
iam_df["IAM_CarbonDensity"] = pd.to_numeric(iam_df["IAM_CarbonDensity"], errors="coerce")


# -------------------------------------------------
# Merge and compute scaling factor
# -------------------------------------------------
merged = esm_df.merge(
    iam_df[["IAM_model", "SSP_std", "IAM_CarbonDensity"]],
    on="SSP_std",
    how="inner",
)

if merged.empty:
    raise ValueError(
        "Merge returned zero rows. Check SSP labels between ESM and IAM inputs."
    )

merged["scaling_factor"] = np.where(
    merged["IAM_CarbonDensity"].isna() | (merged["IAM_CarbonDensity"] == 0),
    np.nan,
    merged["CumulativeCarbonDensity_tCO2_per_ha"] / merged["IAM_CarbonDensity"],
)

# Optional sort for readability
sort_cols = [c for c in ["SSP_std", "IAM_model"] if c in merged.columns]
if sort_cols:
    merged = merged.sort_values(sort_cols).reset_index(drop=True)


# -------------------------------------------------
# Save
# -------------------------------------------------
OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
merged.to_excel(OUTPUT_XLSX, index=False)

print(f"Saved scaling factors to {OUTPUT_FILE}")
print(f"Output rows: {len(merged)}")
print("Rows by SSP:")
print(merged["SSP_std"].value_counts(dropna=False))