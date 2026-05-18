import os
import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
INPUT_ESM_NAME = "cumulative_carbon_density_afforestation_regional_ESM_ensemble_paper.xlsx"
INPUT_IAM_NAME = "cumulative_carbon_density_afforestation_regional_IAM_paper.xlsx"

OUTPUT_XLSX_NAME = "scaling_factors_regions_afforestation_paper.xlsx"
OUTPUT_CSV_NAME = "scaling_factors_regions_afforestation_paper.csv"

CANONICAL_SCENARIO = "SSP1-26"

# Region mapping
REGION_MAP = {
    "Africa": "R10AFRICA",
    "Europe": "R10EUROPE",
    "Latin America and Caribbean": "R10LATIN_AM",
    "Middle East": "R10MIDDLE_EAST",
    "North America": "R10NORTH_AM",
    "Eastern Asia": "R10CHINA+",
    "Southern Asia": "R10INDIA+",
    "Asia-Pacific": "R10PAC_OECD",
    "Eurasia": "R10REF_ECON",
    "South-East Asia and developing Pacific": "R10REST_ASIA",
}

# Column names in ESM file
LSM_MODEL_COL = "Model"          # Land Surface Model (e.g., jules)
ESM_MODEL_COL = "ESM"            # GCM / ESM (e.g., mpi-esm1-2-hr)
ESM_SCENARIO_COL = "SSP"
ESM_REGION_COL = "Region"
ESM_TRANSITION_COL = "Transition"
ESM_CARBON_COL = "Carbon_Density_tCO2_per_ha"

# IAM columns
IAM_SCENARIO_COL = "Scenario"
IAM_REGION_COL = "Region"
IAM_CARBON_COL = "Carbon_Density_tCO2_per_ha"
IAM_MODEL_COL = None  # Set to actual column name if IAM has model splits you want to retain (e.g., "Model")

# If IAM has a model column and you want per-IAM model scaling, set this True
PER_IAM_MODEL = False

# Negative handling: 'keep', 'clip_zero', or 'drop'
NEGATIVE_POLICY = "keep"


# --------------- Path resolution (portable, no local hardcoded path) ---------------
def resolve_data_dir(required_files):
    candidates = []

    # Optional override:
    # export PAPER_FILES_DIR=/path/to/paper_files
    env_dir = os.getenv("PAPER_FILES_DIR")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        candidates.append(p)
        candidates.append(p / "paper_files")

    # Notebook/terminal current directory
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.append(cwd / "paper_files")

    # Script directory
    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
        candidates.append(script_dir / "paper_files")
        candidates.append(script_dir.parent / "paper_files")

    # De-duplicate in order
    seen = set()
    unique = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
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

ESM_XLSX = DATA_DIR / INPUT_ESM_NAME
IAM_XLSX = DATA_DIR / INPUT_IAM_NAME

OUTPUT_XLSX = DATA_DIR / OUTPUT_XLSX_NAME
OUTPUT_CSV = DATA_DIR / OUTPUT_CSV_NAME


# --------------- Helpers ---------------
def normalize_esm_scenario(v):
    return CANONICAL_SCENARIO if isinstance(v, str) and v.lower().strip() == "ssp126" else None


def normalize_iam_scenario(v):
    up = str(v).upper()
    return CANONICAL_SCENARIO if ("SSP1" in up and "26" in up) else None


# --------------- Load ---------------
print("Reading ESM input...")
esm = pd.read_excel(ESM_XLSX)
print("Reading IAM input...")
iam = pd.read_excel(IAM_XLSX)

# --------------- Scenario normalize ---------------
esm["SSP_std"] = esm[ESM_SCENARIO_COL].apply(normalize_esm_scenario)
iam["SSP_std"] = iam[IAM_SCENARIO_COL].apply(normalize_iam_scenario)

esm = esm[esm["SSP_std"] == CANONICAL_SCENARIO].copy()
iam = iam[iam["SSP_std"] == CANONICAL_SCENARIO].copy()

print("\nAfter scenario filter:")
print("  ESM rows:", len(esm))
print("  IAM rows:", len(iam))

# --------------- Region mapping ---------------
esm = esm[esm[ESM_REGION_COL] != "World"].copy()
esm["Region_mapped"] = esm[ESM_REGION_COL].map(REGION_MAP)

unmapped_regions = esm[esm["Region_mapped"].isna()][ESM_REGION_COL].unique()
if len(unmapped_regions) > 0:
    print("WARNING: Unmapped ESM regions (dropping):", unmapped_regions)
    esm = esm[esm["Region_mapped"].notna()].copy()

# --------------- Negative handling ---------------
neg_mask = esm[ESM_CARBON_COL] < 0
neg_count = neg_mask.sum()
if neg_count > 0:
    print(
        f"\nNOTE: {neg_count} ESM rows have negative carbon density "
        f"(e.g., min={esm[ESM_CARBON_COL].min():.3f}). Policy: {NEGATIVE_POLICY}"
    )
    if NEGATIVE_POLICY == "clip_zero":
        esm.loc[neg_mask, ESM_CARBON_COL] = 0.0
    elif NEGATIVE_POLICY == "drop":
        esm = esm.loc[~neg_mask].copy()

# --------------- Prepare IAM regions ---------------
iam = iam.rename(columns={IAM_REGION_COL: "Region_mapped"})

if IAM_MODEL_COL and PER_IAM_MODEL:
    # keep IAM model column
    pass
else:
    # aggregate across IAM model dimension if present (mean)
    group_keys = ["SSP_std", "Region_mapped"]
    iam = (
        iam.groupby(group_keys, as_index=False)
           .agg({IAM_CARBON_COL: "mean"})
    )

# --------------- ESM granularity aggregation ---------------
# If there are duplicate rows for same (SSP_std, ESM, LSM, Transition, Region_mapped), average them
esm_grouped = (
    esm.groupby(
        ["SSP_std", ESM_MODEL_COL, LSM_MODEL_COL, ESM_TRANSITION_COL, "Region_mapped"],
        as_index=False
    )
    .agg({ESM_CARBON_COL: "mean"})
    .rename(columns={ESM_CARBON_COL: "ESM_CarbonDensity"})
)

# --------------- IAM grouped ---------------
if IAM_MODEL_COL and PER_IAM_MODEL:
    iam_grouped = (
        iam.groupby(["SSP_std", IAM_MODEL_COL, "Region_mapped"], as_index=False)
           .agg({IAM_CARBON_COL: "mean"})
           .rename(columns={IAM_CARBON_COL: "IAM_CarbonDensity", IAM_MODEL_COL: "IAM_Model"})
    )
else:
    iam_grouped = iam.rename(columns={IAM_CARBON_COL: "IAM_CarbonDensity"})

print("\nESM grouped rows:", len(esm_grouped))
print("IAM grouped rows:", len(iam_grouped))

# --------------- Merge ---------------
if IAM_MODEL_COL and PER_IAM_MODEL:
    merged = esm_grouped.merge(
        iam_grouped,
        on=["SSP_std", "Region_mapped"],
        how="inner",
        validate="many_to_many"
    )
else:
    merged = esm_grouped.merge(
        iam_grouped,
        on=["SSP_std", "Region_mapped"],
        how="inner",
        validate="many_to_one"
    )

# --------------- Scaling ---------------
merged = merged[merged["IAM_CarbonDensity"] != 0].copy()
merged["scaling_factor"] = merged["ESM_CarbonDensity"] / merged["IAM_CarbonDensity"]

# Column ordering
cols = [
    "SSP_std", "Region_mapped", ESM_MODEL_COL, LSM_MODEL_COL, ESM_TRANSITION_COL,
    "ESM_CarbonDensity", "IAM_CarbonDensity", "scaling_factor"
]
if "IAM_Model" in merged.columns:
    cols.insert(5, "IAM_Model")

merged = (
    merged[cols]
    .sort_values(["SSP_std", ESM_MODEL_COL, LSM_MODEL_COL, ESM_TRANSITION_COL, "Region_mapped"])
    .reset_index(drop=True)
)

print("\nPreview (first 15 rows):")
print(merged.head(15))

# Coverage diagnostics
expected_keys = esm_grouped[["SSP_std", ESM_MODEL_COL, LSM_MODEL_COL, ESM_TRANSITION_COL, "Region_mapped"]].drop_duplicates()
matched_keys = merged[["SSP_std", ESM_MODEL_COL, LSM_MODEL_COL, ESM_TRANSITION_COL, "Region_mapped"]].drop_duplicates()
missing = (
    expected_keys
    .merge(matched_keys, how="left", indicator=True)
    .query('_merge == "left_only"')
    .drop(columns="_merge")
)

if not missing.empty:
    print("\nWARNING: Missing IAM matches for some ESM combinations:")
    print(missing.head(15))
    print("Total missing combinations:", len(missing))
else:
    print("\nAll ESM combinations matched IAM data (after filtering).")

# --------------- Save ---------------
OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
merged.to_excel(OUTPUT_XLSX, index=False)
merged.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved scaling factors to: {OUTPUT_XLSX_NAME} and {OUTPUT_CSV_NAME}")

# --------------- Stats ---------------
print("\nScaling factor stats:")
print(merged["scaling_factor"].describe())

# Simple outlier flag (> 99th or < 1st percentile)
low_thr = merged["scaling_factor"].quantile(0.01)
high_thr = merged["scaling_factor"].quantile(0.99)
outliers = merged[(merged["scaling_factor"] < low_thr) | (merged["scaling_factor"] > high_thr)]

if not outliers.empty:
    print(f"\nPotential extreme outliers outside 1st-99th percentile ({len(outliers)} rows). Examples:")
    print(outliers.head(10))