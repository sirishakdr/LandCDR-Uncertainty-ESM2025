import os
import pandas as pd
from pathlib import Path

# ---------------- CONFIG ----------------
INPUT_ESM_NAME = "cumulative_carbon_density_biomass_regional_ESM_ensemble_paper.xlsx"
INPUT_IAM_NAME = "cumulative_carbon_density_biomass_regional_IAM_paper.xlsx"

OUTPUT_XLSX_NAME = "scaling_factors_regions_biomass_ssp126D_paper.xlsx"
OUTPUT_CSV_NAME = "scaling_factors_regions_biomass_ssp126D_paper.csv"

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
ESM_TRANSITION_COL = "Landuse"
ESM_CARBON_COL = "Carbon_Density_tCO2_per_ha"

# IAM optional model split
IAM_MODEL_COL = None   # e.g., "model" if present and needed
PER_IAM_MODEL = False

# Negative handling: keep, clip_zero, or drop
NEGATIVE_POLICY = "keep"


# --------------- Path resolution (portable + notebook-safe, no personal path) ---------------
def resolve_data_dir(required_files):
    candidates = []

    # Optional override for collaborators/CI:
    # export PAPER_FILES_DIR=/path/to/paper_files
    env_dir = os.getenv("PAPER_FILES_DIR")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        candidates.append(p)
        candidates.append(p / "paper_files")

    # Current working directory (notebook/terminal)
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.append(cwd / "paper_files")

    # Script directory (when run as .py)
    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
        candidates.append(script_dir / "paper_files")
        candidates.append(script_dir.parent / "paper_files")

    # De-duplicate in order
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for d in unique_candidates:
        if all((d / f).exists() for f in required_files):
            return d

    checked = "\n".join(str(d) for d in unique_candidates)
    raise FileNotFoundError(
        "Missing required input files.\n"
        "Set PAPER_FILES_DIR or run from a folder containing paper_files.\n"
        "Expected files:\n"
        + "\n".join(f"  - {f}" for f in required_files)
        + "\nChecked directories:\n"
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
    up = str(v).upper().replace("_", "").replace("-", "").replace(".", "")
    return CANONICAL_SCENARIO if ("SSP1" in up and "26" in up) else None

def find_col(df, candidates, required=True, label="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find {label}. Tried: {candidates}. Available: {list(df.columns)}")
    return None


# --------------- Load ---------------
print("Data directory detected successfully.")
print("Reading ESM input file...")
esm = pd.read_excel(ESM_XLSX)
print("Reading IAM input file...")
iam = pd.read_excel(IAM_XLSX)

print("\nESM columns:", list(esm.columns))
print("IAM columns:", list(iam.columns))


# --------------- Resolve IAM columns robustly ---------------
IAM_SCENARIO_COL = find_col(
    iam,
    ["scenario", "Scenario", "SCENARIO", "ssp", "SSP"],
    required=True,
    label="IAM scenario column"
)

IAM_REGION_COL = find_col(
    iam,
    ["region", "Region", "REGION", "Region_mapped"],
    required=True,
    label="IAM region column"
)

IAM_CARBON_COL = find_col(
    iam,
    [
        "Carbon_Density_tCO2_per_ha",
        "carbon_density_agr_crops",
        "Cumulative_Carbon_Density_2020_2100_tCO2_ha",
        "value",
        "Value",
    ],
    required=False,
    label="IAM carbon column"
)

# If not found by name, choose first numeric column not obviously year/id
if IAM_CARBON_COL is None:
    numeric_cols = iam.select_dtypes(include="number").columns.tolist()
    filtered_numeric = [
        c for c in numeric_cols
        if not (str(c).isdigit() and 1800 <= int(c) <= 2200)
        and not str(c).lower().startswith("unnamed")
    ]
    if len(filtered_numeric) == 1:
        IAM_CARBON_COL = filtered_numeric[0]
    elif len(filtered_numeric) > 1:
        raise KeyError(
            f"Could not uniquely infer IAM carbon column. Numeric candidates: {filtered_numeric}. "
            "Set IAM_CARBON_COL explicitly."
        )
    else:
        raise KeyError("No suitable numeric IAM carbon column found.")

print("\nDetected IAM columns:")
print("  scenario:", IAM_SCENARIO_COL)
print("  region  :", IAM_REGION_COL)
print("  carbon  :", IAM_CARBON_COL)


# --------------- Scenario normalize ---------------
esm["SSP_std"] = esm[ESM_SCENARIO_COL].apply(normalize_esm_scenario)
iam["SSP_std"] = iam[IAM_SCENARIO_COL].apply(normalize_iam_scenario)

esm = esm[esm["SSP_std"] == CANONICAL_SCENARIO].copy()
iam = iam[iam["SSP_std"] == CANONICAL_SCENARIO].copy()

print("\nAfter scenario filter:")
print("  ESM rows:", len(esm))
print("  IAM rows:", len(iam))

if esm.empty:
    raise ValueError("No ESM rows after scenario filter.")
if iam.empty:
    raise ValueError("No IAM rows after scenario filter.")


# --------------- Region mapping ---------------
esm = esm[esm[ESM_REGION_COL] != "World"].copy()
esm["Region_mapped"] = esm[ESM_REGION_COL].map(REGION_MAP)

unmapped_regions = esm.loc[esm["Region_mapped"].isna(), ESM_REGION_COL].dropna().unique()
if len(unmapped_regions) > 0:
    print("WARNING: Unmapped ESM regions (dropping):", unmapped_regions)
    esm = esm[esm["Region_mapped"].notna()].copy()


# --------------- Negative handling ---------------
neg_mask = esm[ESM_CARBON_COL] < 0
neg_count = int(neg_mask.sum())
if neg_count > 0:
    print(
        f"\nNOTE: {neg_count} ESM rows have negative carbon density "
        f"(min={esm[ESM_CARBON_COL].min():.3f}). Policy: {NEGATIVE_POLICY}"
    )
    if NEGATIVE_POLICY == "clip_zero":
        esm.loc[neg_mask, ESM_CARBON_COL] = 0.0
    elif NEGATIVE_POLICY == "drop":
        esm = esm.loc[~neg_mask].copy()


# --------------- Prepare IAM regions ---------------
iam = iam.rename(columns={IAM_REGION_COL: "Region_mapped"})

# If IAM has verbose region names, map them too (safe no-op if already R10*)
if iam["Region_mapped"].dtype == object:
    iam["Region_mapped"] = iam["Region_mapped"].replace(REGION_MAP)

iam = iam[iam["Region_mapped"].notna()].copy()

# Ensure numeric IAM carbon
iam[IAM_CARBON_COL] = pd.to_numeric(iam[IAM_CARBON_COL], errors="coerce")
iam = iam[iam[IAM_CARBON_COL].notna()].copy()

if IAM_MODEL_COL and PER_IAM_MODEL and IAM_MODEL_COL in iam.columns:
    pass
else:
    group_keys = ["SSP_std", "Region_mapped"]
    iam = (
        iam.groupby(group_keys, as_index=False)
           .agg({IAM_CARBON_COL: "mean"})
    )


# --------------- ESM granularity aggregation ---------------
esm_grouped = (
    esm.groupby(["SSP_std", ESM_MODEL_COL, LSM_MODEL_COL, ESM_TRANSITION_COL, "Region_mapped"], as_index=False)
       .agg({ESM_CARBON_COL: "mean"})
       .rename(columns={ESM_CARBON_COL: "ESM_CarbonDensity"})
)

# --------------- IAM grouped ---------------
if IAM_MODEL_COL and PER_IAM_MODEL and IAM_MODEL_COL in iam.columns:
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
if IAM_MODEL_COL and PER_IAM_MODEL and "IAM_Model" in iam_grouped.columns:
    merged = esm_grouped.merge(
        iam_grouped, on=["SSP_std", "Region_mapped"], how="inner", validate="many_to_many"
    )
else:
    merged = esm_grouped.merge(
        iam_grouped, on=["SSP_std", "Region_mapped"], how="inner", validate="many_to_one"
    )


# --------------- Scaling ---------------
merged = merged[merged["IAM_CarbonDensity"] != 0].copy()
merged["scaling_factor"] = merged["ESM_CarbonDensity"] / merged["IAM_CarbonDensity"]

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
print("\nSaved scaling factors in data directory:")
print(" ", OUTPUT_XLSX_NAME)
print(" ", OUTPUT_CSV_NAME)

# --------------- Stats ---------------
print("\nScaling factor stats:")
print(merged["scaling_factor"].describe())

low_thr = merged["scaling_factor"].quantile(0.01)
high_thr = merged["scaling_factor"].quantile(0.99)
outliers = merged[(merged["scaling_factor"] < low_thr) | (merged["scaling_factor"] > high_thr)]
if not outliers.empty:
    print(f"\nPotential extreme outliers outside 1st-99th percentile ({len(outliers)} rows). Examples:")
    print(outliers.head(10))