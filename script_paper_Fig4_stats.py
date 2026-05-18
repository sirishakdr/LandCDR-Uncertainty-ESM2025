#!/usr/bin/env python3
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyam
from matplotlib.gridspec import GridSpec
from scipy import stats
import matplotlib.patches as mpatches

# --- Configuration ---
SCENARIO = "SSP1-26"
REGION = "World"
VAR_AFF_CO2 = "Carbon Sequestration|Land Use|Afforestation"
YEAR_START = 2020
YEAR_END = 2100
YEAR_MAX = 2100

MODELS_ORDER = [
    "AIM/CGE 2.0",
    "GCAM 4.2",
    "IMAGE 3.0.1",
    "MESSAGE-GLOBIOM 1.0",
    "REMIND-MAGPIE 1.5",
]

NET_ZERO_YEARS = {
    "AIM/CGE 2.0": 2100,
    "GCAM 4.2": 2079,
    "IMAGE 3.0.1": 2076,
    "MESSAGE-GLOBIOM 1.0": 2073,
    "REMIND-MAGPIE 1.5": 2075,
}

AR6_WORLD_CSV_NAME = "AR6_Scenarios_Database_World_v1.1.csv"
META_XLSX_NAME = "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
META_SHEET = "meta_Ch3vetted_withclimate"
SF_FILE_NAME = "scaling_factors_afforestation_global_paper.xlsx"

OUTPUT_FIG_BASE_NAME = "paper_Fig4"
OUTPUT_STATS_XLSX_NAME = "paper_Fig4_stats.xlsx"


def resolve_data_dir(required_files):
    candidates = []

    env_dir = os.getenv("PAPER_FILES_DIR")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        candidates.append(p)
        candidates.append(p / "paper_files")

    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.append(cwd / "paper_files")

    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
        candidates.append(script_dir / "paper_files")
        candidates.append(script_dir.parent / "paper_files")

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

    checked = "\n".join(str(d) for d in unique_candidates)
    raise FileNotFoundError(
        "Could not locate required input files.\n"
        "Set PAPER_FILES_DIR or run from a folder containing paper_files.\n"
        "Expected files:\n"
        + "\n".join(f"  - {f}" for f in required_files)
        + "\nChecked locations:\n"
        + checked
    )


DATA_DIR = resolve_data_dir([AR6_WORLD_CSV_NAME, META_XLSX_NAME, SF_FILE_NAME])

AR6_WORLD_CSV = DATA_DIR / AR6_WORLD_CSV_NAME
META_XLSX = DATA_DIR / META_XLSX_NAME
SF_FILE = DATA_DIR / SF_FILE_NAME

OUTPUT_FIG_PNG = DATA_DIR / f"{OUTPUT_FIG_BASE_NAME}.png"
OUTPUT_FIG_PDF = DATA_DIR / f"{OUTPUT_FIG_BASE_NAME}.pdf"
OUTPUT_STATS_XLSX = DATA_DIR / OUTPUT_STATS_XLSX_NAME


def _norm_key(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _pick_col(df, aliases, required=True, label="column"):
    col_map = {_norm_key(c): c for c in df.columns}
    for a in aliases:
        k = _norm_key(a)
        if k in col_map:
            return col_map[k]
    if required:
        raise ValueError(f"Missing {label}. Tried aliases: {aliases}. Available: {list(df.columns)}")
    return None


def normalize_transition(v):
    s = _norm_key(v)
    if s in {"nattoaff", "naturaltoafforestation", "naturaltoaff", "nat2aff"}:
        return "nattoaff"
    if s in {"agtoaff", "agriculturaltoafforestation", "agriculturaltoaff", "ag2aff"}:
        return "agtoaff"
    if s in {"agtonat", "agriculturaltonatural", "agtonatural"}:
        return "agtonat"
    return s


def get_years_from_columns(df, year_max=YEAR_MAX):
    years = []
    for c in df.columns:
        if isinstance(c, (int, np.integer)):
            if c <= year_max:
                years.append(int(c))
        elif isinstance(c, str) and c.isdigit():
            ci = int(c)
            if ci <= year_max:
                years.append(ci)
    return np.array(sorted(set(years)), dtype=int)


def row_to_year_series(row, years):
    vals = []
    for y in years:
        if y in row.index:
            vals.append(row[y])
        elif str(y) in row.index:
            vals.append(row[str(y)])
        else:
            vals.append(np.nan)
    return pd.Series(vals, index=years, dtype=float)


def cumulative_mt(series: pd.Series, first_year: int, last_year: int) -> float:
    # pyam cumulative integration with guard for empty windows
    if int(first_year) > int(last_year):
        return 0.0
    val = pyam.timeseries.cumulative(series, int(first_year), int(last_year))
    if val is None or not np.isfinite(val):
        return np.nan
    return float(val)


def q(arr, p):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.percentile(arr, p))


# --- Load scaling factors ---
sf_df = pd.read_excel(SF_FILE)

print("Column names in scaling factors file:")
print(sf_df.columns.tolist())

sf_df.columns = [str(c).strip() for c in sf_df.columns]

col_iam = _pick_col(sf_df, ["IAM_model", "iammodel", "iam", "model_iam"], required=True, label="IAM model")
col_lsm = _pick_col(sf_df, ["Model", "LSM", "LandModel", "land_model"], required=False, label="LSM")
col_esm = _pick_col(sf_df, ["ESM", "esm"], required=False, label="ESM")
col_trans = _pick_col(
    sf_df,
    ["Transition", "transition", "Landuse", "landuse", "land_use", "land_transition"],
    required=False,
    label="transition",
)
col_sf = _pick_col(sf_df, ["scaling_factor", "scalingfactor", "sf"], required=True, label="scaling factor")

sf_df["IAM_model"] = sf_df[col_iam].astype(str).str.strip()
sf_df["IAM_model_lc"] = sf_df["IAM_model"].str.lower()
sf_df["LSM"] = sf_df[col_lsm].astype(str).str.strip().str.lower() if col_lsm else "unknown"
sf_df["ESM"] = sf_df[col_esm].astype(str).str.strip().str.lower() if col_esm else "unknown"
sf_df["Transition"] = sf_df[col_trans].astype(str).str.strip().map(normalize_transition) if col_trans else "unknown"
sf_df["scaling_factor"] = pd.to_numeric(sf_df[col_sf], errors="coerce")

# *** FILTER OUT JULES ***

print(f"Original number of rows: {len(sf_df)}")
sf_df = sf_df[sf_df["LSM"] != "jules"].copy()
print(f"After filtering JULES: {len(sf_df)}")

# *** FILTER OUT AGTONAT ***

print(f"Before AgToNat filter: {len(sf_df)}")
sf_df = sf_df[sf_df["Transition"] != "agtonat"].copy()
print(f"After filtering AgToNat: {len(sf_df)}")

sf_df = sf_df[np.isfinite(sf_df["scaling_factor"])].copy()

print("\nUnique LSMs:", sorted(sf_df["LSM"].dropna().unique()))
print("Unique ESMs:", sorted(sf_df["ESM"].dropna().unique()))


# --- Load & compute unscaled ---
def compute_unscaled_aff_df():
    data = pyam.IamDataFrame(data=str(AR6_WORLD_CSV))
    meta = pd.read_excel(META_XLSX, sheet_name=META_SHEET)
    meta = meta.rename(columns={"Model": "model", "Scenario": "scenario"})
    data.set_meta(meta=meta.set_index(["model", "scenario"]))

    if SCENARIO not in set(data.scenario):
        raise ValueError(f"Scenario '{SCENARIO}' not found.")

    filtered = data.filter(scenario=SCENARIO, region=REGION)
    var = filtered.filter(variable=VAR_AFF_CO2, year=range(YEAR_START, YEAR_END + 1))
    if var.data.empty:
        raise ValueError("No afforestation data found for selected scenario/region/variable.")

    decadal_years = sorted({int(y) for y in var.data["year"].unique() if int(y) % 10 == 0})
    var = var.filter(year=decadal_years)

    aff_df = var.timeseries().reset_index()
    year_cols = [c for c in aff_df.columns if isinstance(c, (int, np.integer)) or (isinstance(c, str) and c.isdigit())]
    aff_df[year_cols] = aff_df[year_cols].apply(pd.to_numeric, errors="coerce")

    return aff_df.set_index(["model", "scenario", "region", "variable", "unit"])


Afforestation_total_df = compute_unscaled_aff_df()

# --- Prepare data ---
df_models = Afforestation_total_df.index.get_level_values("model").unique()
lower_to_df_model = {m.lower(): m for m in df_models}
years = get_years_from_columns(Afforestation_total_df, YEAR_END)

print("\nIntegration check: using pyam.timeseries.cumulative with non-overlapping windows.")
print("Up-to window: [2020, NZ], Post window: [NZ+1, 2100]")

# --- Compute model-level cumulative checks (full, up-to, post non-overlap) ---
model_cumulative = {}
split_check_rows = []

for model_key in MODELS_ORDER:
    model_lc = model_key.strip().lower()
    df_model_name = lower_to_df_model.get(model_lc)
    if not df_model_name:
        continue

    nz_year = int(NET_ZERO_YEARS.get(model_key, YEAR_MAX))

    try:
        sel = Afforestation_total_df.loc[(df_model_name, SCENARIO, REGION, VAR_AFF_CO2)]
        row = sel.iloc[0] if isinstance(sel, pd.DataFrame) else sel
        sequestration_series = row_to_year_series(row, years)
    except KeyError:
        continue

    full_mt = cumulative_mt(sequestration_series, YEAR_START, YEAR_END)
    upto_mt = cumulative_mt(sequestration_series, YEAR_START, nz_year)
    post_mt = cumulative_mt(sequestration_series, nz_year + 1, YEAR_END)

    full_gt = full_mt / 1000.0 if np.isfinite(full_mt) else np.nan
    upto_gt = upto_mt / 1000.0 if np.isfinite(upto_mt) else np.nan
    post_gt = post_mt / 1000.0 if np.isfinite(post_mt) else np.nan

    split_sum_gt = (upto_gt + post_gt) if np.isfinite(upto_gt) and np.isfinite(post_gt) else np.nan
    delta_gt = (split_sum_gt - full_gt) if np.isfinite(split_sum_gt) and np.isfinite(full_gt) else np.nan

    model_cumulative[model_key] = {
        "NZ_Year": nz_year,
        "Full_2020_2100_GtCO2": full_gt,
        "UpTo_NZ_GtCO2": upto_gt,
        "Post_NZplus1_2100_GtCO2": post_gt,
        "SplitSum_GtCO2": split_sum_gt,
        "Delta_GtCO2": delta_gt,
    }

    split_check_rows.append(
        {
            "IAM": model_key,
            "NZ_Year": nz_year,
            "Full_2020_2100_GtCO2": full_gt,
            "UpTo_NZ_GtCO2": upto_gt,
            "Post_NZplus1_2100_GtCO2": post_gt,
            "SplitSum_GtCO2": split_sum_gt,
            "Delta_GtCO2": delta_gt,
            "Abs_Delta_GtCO2": abs(delta_gt) if np.isfinite(delta_gt) else np.nan,
        }
    )

split_check_df = pd.DataFrame(split_check_rows)
if not split_check_df.empty:
    print("\nSplit consistency (GtCO2):")
    print(split_check_df[["IAM", "NZ_Year", "Delta_GtCO2", "Abs_Delta_GtCO2"]].to_string(index=False))
    print("\nDelta summary:")
    print(split_check_df["Delta_GtCO2"].describe())


# --- Create violin plot data (UP TO NET-ZERO) ---
violin_data = []
unscaled_values = {}

for model_key in MODELS_ORDER:
    info = model_cumulative.get(model_key)
    if info is None:
        continue

    model_lc = model_key.strip().lower()
    cum_gt = info["UpTo_NZ_GtCO2"]

    if not np.isfinite(cum_gt):
        continue

    unscaled_values[model_key] = cum_gt
    sf_rows = sf_df[sf_df["IAM_model_lc"] == model_lc]

    for _, row in sf_rows.iterrows():
        scaled_carbon = cum_gt * row["scaling_factor"]
        if np.isfinite(scaled_carbon):
            violin_data.append(
                {
                    "IAM": model_key,
                    "LSM": row["LSM"],
                    "ScaledCarbon": scaled_carbon,
                    "UnscaledCarbon": cum_gt,
                    "ScalingFactor": row["scaling_factor"],
                    "Transition": row["Transition"],
                    "ESM": row["ESM"],
                }
            )

df_violin = pd.DataFrame(violin_data)
print(f"\n*** Up-to Net-Zero data points (excluding JULES & AgToNat): {len(df_violin)} ***")

# --- Create violin plot data (POST NET-ZERO, non-overlapping from NZ+1) ---
violin_data_post_nz = []
unscaled_values_post_nz = {}

for model_key in MODELS_ORDER:
    info = model_cumulative.get(model_key)
    if info is None:
        continue

    model_lc = model_key.strip().lower()
    cum_gt_post_nz = info["Post_NZplus1_2100_GtCO2"]

    if not np.isfinite(cum_gt_post_nz):
        continue

    unscaled_values_post_nz[model_key] = cum_gt_post_nz
    sf_rows = sf_df[sf_df["IAM_model_lc"] == model_lc]

    for _, row in sf_rows.iterrows():
        scaled_carbon_post_nz = cum_gt_post_nz * row["scaling_factor"]
        if np.isfinite(scaled_carbon_post_nz):
            violin_data_post_nz.append(
                {
                    "IAM": model_key,
                    "LSM": row["LSM"],
                    "ScaledCarbon": scaled_carbon_post_nz,
                    "UnscaledCarbon": cum_gt_post_nz,
                    "ScalingFactor": row["scaling_factor"],
                    "Transition": row["Transition"],
                    "ESM": row["ESM"],
                }
            )

df_violin_post_nz = pd.DataFrame(violin_data_post_nz)
print(f"*** Post-Net-Zero data points (excluding JULES & AgToNat): {len(df_violin_post_nz)} ***\n")

# Filter models with any data
models_with_data = [
    model
    for model in MODELS_ORDER
    if ((not df_violin.empty and model in set(df_violin["IAM"])) or
        (not df_violin_post_nz.empty and model in set(df_violin_post_nz["IAM"])))
]
if not models_with_data:
    raise ValueError("No models with valid scaled data after filtering.")

lsm_set = set()
if not df_violin.empty:
    lsm_set.update(df_violin["LSM"].dropna().unique())
if not df_violin_post_nz.empty:
    lsm_set.update(df_violin_post_nz["LSM"].dropna().unique())
lsm_order = sorted(lsm_set)

print(f"*** LSMs included in analysis: {lsm_order} ***")


# --- Variance Decomposition ---
def variance_decomposition_factorial(df):
    results = []

    for iam in models_with_data:
        if iam not in set(df["IAM"]):
            continue

        iam_data = df[df["IAM"] == iam].copy()
        values = iam_data["ScaledCarbon"].values
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue

        grand_mean = values.mean()
        n = len(values)

        SS_total = np.sum((values - grand_mean) ** 2)
        if SS_total == 0:
            continue

        lsm_levels = sorted(iam_data["LSM"].dropna().unique())
        esm_levels = sorted(iam_data["ESM"].dropna().unique())
        n_lsm = len(lsm_levels)
        n_esm = len(esm_levels)

        if n_lsm == 0 or n_esm == 0:
            continue

        lsm_means = {}
        lsm_counts = {}
        for lsm in lsm_levels:
            lsm_data = iam_data[iam_data["LSM"] == lsm]["ScaledCarbon"].values
            lsm_data = lsm_data[np.isfinite(lsm_data)]
            if len(lsm_data) == 0:
                continue
            lsm_means[lsm] = lsm_data.mean()
            lsm_counts[lsm] = len(lsm_data)

        esm_means = {}
        esm_counts = {}
        for esm in esm_levels:
            esm_data = iam_data[iam_data["ESM"] == esm]["ScaledCarbon"].values
            esm_data = esm_data[np.isfinite(esm_data)]
            if len(esm_data) == 0:
                continue
            esm_means[esm] = esm_data.mean()
            esm_counts[esm] = len(esm_data)

        cell_means = {}
        cell_counts = {}
        for lsm in lsm_levels:
            for esm in esm_levels:
                cell_data = iam_data[(iam_data["LSM"] == lsm) & (iam_data["ESM"] == esm)]["ScaledCarbon"].values
                cell_data = cell_data[np.isfinite(cell_data)]
                if len(cell_data) > 0:
                    cell_means[(lsm, esm)] = cell_data.mean()
                    cell_counts[(lsm, esm)] = len(cell_data)

        SS_LSM = sum(lsm_counts[lsm] * (lsm_means[lsm] - grand_mean) ** 2 for lsm in lsm_means)
        SS_ESM = sum(esm_counts[esm] * (esm_means[esm] - grand_mean) ** 2 for esm in esm_means)

        SS_interaction = 0.0
        for (lsm, esm), cell_mean in cell_means.items():
            n_cell = cell_counts[(lsm, esm)]
            expected_additive = lsm_means[lsm] + esm_means[esm] - grand_mean
            interaction_effect = cell_mean - expected_additive
            SS_interaction += n_cell * (interaction_effect ** 2)

        SS_residual = 0.0
        for (lsm, esm), cell_mean in cell_means.items():
            cell_data = iam_data[(iam_data["LSM"] == lsm) & (iam_data["ESM"] == esm)]["ScaledCarbon"].values
            cell_data = cell_data[np.isfinite(cell_data)]
            SS_residual += np.sum((cell_data - cell_mean) ** 2)

        var_lsm_pct = (SS_LSM / SS_total) * 100.0
        var_esm_pct = (SS_ESM / SS_total) * 100.0
        var_interaction_pct = (SS_interaction / SS_total) * 100.0
        var_residual_pct = (SS_residual / SS_total) * 100.0

        df_lsm = n_lsm - 1
        df_esm = n_esm - 1
        df_interaction = (n_lsm - 1) * (n_esm - 1)
        df_residual = n - n_lsm * n_esm

        MS_LSM = SS_LSM / df_lsm if df_lsm > 0 else np.nan
        MS_ESM = SS_ESM / df_esm if df_esm > 0 else np.nan
        MS_interaction = SS_interaction / df_interaction if df_interaction > 0 else np.nan
        MS_residual = SS_residual / df_residual if df_residual > 0 else np.nan

        F_LSM = MS_LSM / MS_residual if np.isfinite(MS_residual) and MS_residual > 0 else np.nan
        F_ESM = MS_ESM / MS_residual if np.isfinite(MS_residual) and MS_residual > 0 else np.nan
        F_interaction = MS_interaction / MS_residual if np.isfinite(MS_residual) and MS_residual > 0 else np.nan

        p_LSM = 1 - stats.f.cdf(F_LSM, df_lsm, df_residual) if np.isfinite(F_LSM) else np.nan
        p_ESM = 1 - stats.f.cdf(F_ESM, df_esm, df_residual) if np.isfinite(F_ESM) else np.nan
        p_interaction = 1 - stats.f.cdf(F_interaction, df_interaction, df_residual) if np.isfinite(F_interaction) else np.nan

        results.append(
            {
                "IAM": iam,
                "Var_LSM": var_lsm_pct,
                "Var_ESM": var_esm_pct,
                "Var_Interaction": var_interaction_pct,
                "Var_Residual": var_residual_pct,
                "F_LSM": F_LSM,
                "F_ESM": F_ESM,
                "F_Interaction": F_interaction,
                "p_LSM": p_LSM,
                "p_ESM": p_ESM,
                "p_Interaction": p_interaction,
            }
        )

    return pd.DataFrame(results)


variance_df = variance_decomposition_factorial(df_violin)
variance_df_post_nz = variance_decomposition_factorial(df_violin_post_nz)

# Preserve original behavior: use first row if available
universal_var = {
    "LSM": variance_df["Var_LSM"].iloc[0] if len(variance_df) > 0 else 0,
    "ESM": variance_df["Var_ESM"].iloc[0] if len(variance_df) > 0 else 0,
    "Interaction": variance_df["Var_Interaction"].iloc[0] if len(variance_df) > 0 else 0,
    "Residual": variance_df["Var_Residual"].iloc[0] if len(variance_df) > 0 else 0,
}

universal_var_post_nz = {
    "LSM": variance_df_post_nz["Var_LSM"].iloc[0] if len(variance_df_post_nz) > 0 else 0,
    "ESM": variance_df_post_nz["Var_ESM"].iloc[0] if len(variance_df_post_nz) > 0 else 0,
    "Interaction": variance_df_post_nz["Var_Interaction"].iloc[0] if len(variance_df_post_nz) > 0 else 0,
    "Residual": variance_df_post_nz["Var_Residual"].iloc[0] if len(variance_df_post_nz) > 0 else 0,
}


def summarize_period_stats(df_period, unscaled_map, period_label):
    rows = []
    trans_labels = {
        "nattoaff": "Natural to Afforestation",
        "agtoaff": "Agricultural to Afforestation",
    }

    for iam in MODELS_ORDER:
        if iam not in unscaled_map:
            continue

        iam_data = df_period[df_period["IAM"] == iam]
        if iam_data.empty:
            continue

        unscaled = float(unscaled_map[iam])
        nz_year = NET_ZERO_YEARS.get(iam, np.nan)

        for t_key, t_label in trans_labels.items():
            sub = iam_data[iam_data["Transition"] == t_key]["ScaledCarbon"].to_numpy(dtype=float)
            sub = sub[np.isfinite(sub)]
            if sub.size == 0:
                continue

            ratios = np.array([unscaled / abs(s) for s in sub if s != 0], dtype=float)
            ratios = ratios[np.isfinite(ratios)]

            rows.append(
                {
                    "Period": period_label,
                    "IAM": iam,
                    "NZ_Year": nz_year,
                    "LUC": t_label,
                    "Unscaled_GtCO2": unscaled,
                    "Scaled_Median_GtCO2": float(np.median(sub)),
                    "Scaled_Q25_GtCO2": q(sub, 25),
                    "Scaled_Q75_GtCO2": q(sub, 75),
                    "Overestimation_Median_times": float(np.median(ratios)) if ratios.size > 0 else np.nan,
                    "Overestimation_Q25_times": q(ratios, 25),
                    "Overestimation_Q75_times": q(ratios, 75),
                    "N": int(sub.size),
                }
            )

        sub_all = iam_data["ScaledCarbon"].to_numpy(dtype=float)
        sub_all = sub_all[np.isfinite(sub_all)]
        if sub_all.size > 0:
            ratios_all = np.array([unscaled / abs(s) for s in sub_all if s != 0], dtype=float)
            ratios_all = ratios_all[np.isfinite(ratios_all)]

            rows.append(
                {
                    "Period": period_label,
                    "IAM": iam,
                    "NZ_Year": nz_year,
                    "LUC": "All (Combined)",
                    "Unscaled_GtCO2": unscaled,
                    "Scaled_Median_GtCO2": float(np.median(sub_all)),
                    "Scaled_Q25_GtCO2": q(sub_all, 25),
                    "Scaled_Q75_GtCO2": q(sub_all, 75),
                    "Overestimation_Median_times": float(np.median(ratios_all)) if ratios_all.size > 0 else np.nan,
                    "Overestimation_Q25_times": q(ratios_all, 25),
                    "Overestimation_Q75_times": q(ratios_all, 75),
                    "N": int(sub_all.size),
                }
            )

    return pd.DataFrame(rows)


stats_upto_df = summarize_period_stats(df_violin, unscaled_values, "Up-to Net-Zero")
stats_post_df = summarize_period_stats(df_violin_post_nz, unscaled_values_post_nz, "Post Net-Zero")

summary_df = pd.DataFrame(
    [
        {"Metric": "Scenario", "Value": SCENARIO},
        {"Metric": "Region", "Value": REGION},
        {"Metric": "Data points up-to NZ", "Value": len(df_violin)},
        {"Metric": "Data points post NZ+1", "Value": len(df_violin_post_nz)},
        {"Metric": "Models with data", "Value": ", ".join(models_with_data)},
        {"Metric": "LSMs included", "Value": ", ".join(lsm_order)},
        {
            "Metric": "Max abs split delta (GtCO2)",
            "Value": float(split_check_df["Abs_Delta_GtCO2"].max()) if not split_check_df.empty else np.nan,
        },
    ]
)

with pd.ExcelWriter(OUTPUT_STATS_XLSX, engine="openpyxl") as writer:
    stats_upto_df.to_excel(writer, sheet_name="UpToNZ_Stats", index=False)
    stats_post_df.to_excel(writer, sheet_name="PostNZplus1_Stats", index=False)
    variance_df.to_excel(writer, sheet_name="ANOVA_UpToNZ", index=False)
    variance_df_post_nz.to_excel(writer, sheet_name="ANOVA_PostNZplus1", index=False)
    split_check_df.to_excel(writer, sheet_name="SplitConsistency", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

print(f"\nSaved stats workbook: {OUTPUT_STATS_XLSX.name}")


# ==============================================================================
# COMBINED FIGURE: Up-to Net-Zero and Post Net-Zero (WITHOUT JULES & AgToNat)
# ==============================================================================

plt.style.use("default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

fig = plt.figure(figsize=(36, 16))
gs = GridSpec(
    2,
    2,
    figure=fig,
    hspace=0.45,
    wspace=0.28,
    width_ratios=[3.8, 1.2],
    left=0.05,
    right=0.97,
    top=0.94,
    bottom=0.10,
)

# Define afforestation color scheme (EXCLUDING JULES & AgToNat)
lsm_colors = {
    "clm": "#f2c45f",
    "jsbach": "#bc5090",
    "orchidee": "#118ab2",
}
bar_colors_list = ["#e74c3c", "#2c3e50", "#3498db", "#95a5a6"]

# Calculate max values for matching axes
all_scaled_values = []
if not df_violin.empty:
    all_scaled_values.extend(df_violin["ScaledCarbon"].dropna().tolist())
if not df_violin_post_nz.empty:
    all_scaled_values.extend(df_violin_post_nz["ScaledCarbon"].dropna().tolist())

all_unscaled_values = list(unscaled_values.values()) + list(unscaled_values_post_nz.values())
all_values_combined = [v for v in (all_scaled_values + all_unscaled_values) if np.isfinite(v)]

if len(all_values_combined) == 0:
    max_aff_violin = 1.0
    min_aff_violin = 0.0
else:
    max_aff_violin = max(all_values_combined)
    min_aff_violin = min(all_values_combined)

max_aff_violin_y = max_aff_violin * 1.15 if max_aff_violin != 0 else 1.0
min_aff_violin_y = min_aff_violin * 1.15 if min_aff_violin < 0 else -1

variances_all = [
    universal_var["LSM"],
    universal_var["ESM"],
    universal_var["Interaction"],
    universal_var["Residual"],
    universal_var_post_nz["LSM"],
    universal_var_post_nz["ESM"],
    universal_var_post_nz["Interaction"],
    universal_var_post_nz["Residual"],
]
max_aff_variance = max(variances_all) if len(variances_all) > 0 else 0
max_aff_variance_y = max_aff_variance * 1.42 if max_aff_variance > 0 else 1


def plot_violin_and_bar_aff(
    fig,
    gs_row,
    col_offset,
    df_violin_data,
    unscaled_vals,
    variance_df_data,
    universal_var_data,
    title_prefix,
    variance_title,
    panel_start_letter,
    show_legend=False,
    violin_ylim=None,
    variance_ylim=None,
):
    # Violin plot
    ax_violin = fig.add_subplot(gs[gs_row, col_offset])
    ax_violin.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=0.8, zorder=0)
    ax_violin.set_axisbelow(True)
    ax_violin.set_facecolor("#fafafa")

    for iam_idx, iam in enumerate(models_with_data):
        if iam not in set(df_violin_data["IAM"]):
            continue

        iam_data = df_violin_data[df_violin_data["IAM"] == iam]

        n_violins = len(lsm_order) + 1
        violin_width = 0.12
        total_width = 0.9
        spacing = total_width / n_violins
        start_offset = -total_width / 2 + spacing / 2

        for lsm_idx, lsm in enumerate(lsm_order):
            lsm_data = iam_data[iam_data["LSM"] == lsm]["ScaledCarbon"].values
            lsm_data = lsm_data[np.isfinite(lsm_data)]
            if len(lsm_data) == 0:
                continue

            position = iam_idx + start_offset + lsm_idx * spacing

            parts = ax_violin.violinplot(
                [lsm_data],
                positions=[position],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method="scott",
            )

            lsm_color = lsm_colors.get(lsm, "#808080")
            for pc in parts["bodies"]:
                pc.set_facecolor(lsm_color)
                pc.set_alpha(0.65)
                pc.set_edgecolor("white")
                pc.set_linewidth(1)

            q1, median, q3 = np.percentile(lsm_data, [25, 50, 75])
            ax_violin.plot([position, position], [q1, q3], color="#1a1a1a", linewidth=3, alpha=0.85, zorder=10)
            ax_violin.plot([position - 0.08, position + 0.08], [median, median], color="#1a1a1a", linewidth=3, alpha=0.85, zorder=10)

        # Overall violin in red
        all_data = iam_data["ScaledCarbon"].values
        all_data = all_data[np.isfinite(all_data)]
        if len(all_data) > 0:
            overall_position = iam_idx + start_offset + len(lsm_order) * spacing
            parts_overall = ax_violin.violinplot(
                [all_data],
                positions=[overall_position],
                widths=violin_width,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method="scott",
            )

            for pc in parts_overall["bodies"]:
                pc.set_facecolor("#a00000")
                pc.set_alpha(0.65)
                pc.set_edgecolor("white")
                pc.set_linewidth(1)

            q1_overall, median_overall, q3_overall = np.percentile(all_data, [25, 50, 75])
            ax_violin.plot([overall_position, overall_position], [q1_overall, q3_overall], color="#8B0000", linewidth=3, alpha=0.9, zorder=10)
            ax_violin.plot([overall_position - 0.08, overall_position + 0.08], [median_overall, median_overall], color="#8B0000", linewidth=3, alpha=0.9, zorder=10)

        if iam in unscaled_vals and np.isfinite(unscaled_vals[iam]):
            unscaled_value = unscaled_vals[iam]
            diamond_position = iam_idx
            ax_violin.plot(
                diamond_position,
                unscaled_value,
                marker="D",
                markersize=16,
                color="#00C853",
                markerfacecolor="#00E676",
                markeredgecolor="#00691B",
                markeredgewidth=3,
                zorder=15,
            )

    for iam_idx in range(len(models_with_data) - 1):
        separator_x = iam_idx + 0.5
        ax_violin.axvline(x=separator_x, color="#cccccc", linestyle="-", linewidth=2.5, alpha=0.4, zorder=1)

    ax_violin.set_xticks(range(len(models_with_data)))
    ax_violin.set_xticklabels(models_with_data, fontsize=24, rotation=0, ha="center", fontweight="bold")
    ax_violin.set_xlabel("", fontsize=24, fontweight="bold", labelpad=12)
    ax_violin.set_ylabel("Cumulative Afforestation Carbon (GtCO$_2$)", fontsize=24, fontweight="bold", labelpad=12)
    ax_violin.set_title(f"({chr(97 + panel_start_letter)}) {title_prefix}", fontsize=26, fontweight="bold", pad=18, loc="left")

    if violin_ylim:
        ax_violin.set_ylim(violin_ylim)

    if show_legend:
        legend_elements = []
        for lsm in lsm_order:
            lsm_color = lsm_colors.get(lsm, "#808080")
            legend_elements.append(
                mpatches.Patch(facecolor=lsm_color, label=lsm.upper(), alpha=0.7, edgecolor="white", linewidth=1)
            )
        legend_elements.append(mpatches.Patch(facecolor="#a00000", label="Overall", alpha=0.7, edgecolor="white", linewidth=1))
        legend_elements.append(mpatches.Patch(facecolor="#00E676", label="Unscaled", alpha=0.85, edgecolor="#00691B", linewidth=2))

        ax_violin.legend(
            handles=legend_elements,
            loc="upper left",
            frameon=True,
            framealpha=0.98,
            fontsize=22,
            title_fontsize=24,
            edgecolor="#cccccc",
            ncol=3,
            handlelength=2.2,
            handleheight=0.8,
            fancybox=True,
            shadow=True,
        )

    ax_violin.spines["top"].set_visible(False)
    ax_violin.spines["right"].set_visible(False)
    ax_violin.spines["left"].set_linewidth(2.5)
    ax_violin.spines["bottom"].set_linewidth(2.5)
    ax_violin.tick_params(axis="both", labelsize=26, width=2.5, length=6)

    # Bar chart
    ax_var = fig.add_subplot(gs[gs_row, col_offset + 1])
    ax_var.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=0.8, zorder=0)
    ax_var.set_axisbelow(True)
    ax_var.set_facecolor("#fafafa")

    sources = ["LSM", "ESM", "LSMxESM\nInteraction", "LUC\n(Residual)"]
    variances = [
        universal_var_data["LSM"],
        universal_var_data["ESM"],
        universal_var_data["Interaction"],
        universal_var_data["Residual"],
    ]

    p_values = [
        variance_df_data["p_LSM"].iloc[0] if len(variance_df_data) > 0 else np.nan,
        variance_df_data["p_ESM"].iloc[0] if len(variance_df_data) > 0 else np.nan,
        variance_df_data["p_Interaction"].iloc[0] if len(variance_df_data) > 0 else np.nan,
        np.nan,
    ]

    x_positions = np.arange(len(sources))
    for i, (x_pos, var, color) in enumerate(zip(x_positions, variances, bar_colors_list)):
        ax_var.bar(x_pos, var, color=color, alpha=0.88, edgecolor="#1a1a1a", linewidth=1.8, width=0.45)

        ax_var.text(x_pos, var + 3, f"{var:.1f}%", ha="center", va="bottom", fontsize=22, fontweight="bold", color="#1a1a1a")

        if not np.isnan(p_values[i]):
            if p_values[i] < 0.001:
                sig_text = "***"
            elif p_values[i] < 0.01:
                sig_text = "**"
            elif p_values[i] < 0.05:
                sig_text = "*"
            else:
                sig_text = "n.s."

            ax_var.text(
                x_pos,
                var + 9,
                sig_text,
                ha="center",
                va="bottom",
                fontsize=22,
                style="italic",
                fontweight="bold",
                color="#d32f2f",
            )

    ax_var.set_xticks(x_positions)
    ax_var.set_xticklabels(sources, fontsize=24, fontweight="bold")
    ax_var.set_ylabel("Variance (%)", fontsize=24, fontweight="bold", labelpad=10)
    ax_var.set_title(f"({chr(98 + panel_start_letter)}) {variance_title}", fontsize=28, fontweight="bold", pad=15, loc="left")

    if variance_ylim:
        ax_var.set_ylim([-variance_ylim * 0.1, variance_ylim])

    if show_legend:
        legend_patches = [
            mpatches.Patch(color=bar_colors_list[0], label="LSM", alpha=0.88, edgecolor="#1a1a1a", linewidth=1.5),
            mpatches.Patch(color=bar_colors_list[1], label="ESM", alpha=0.88, edgecolor="#1a1a1a", linewidth=1.5),
            mpatches.Patch(color=bar_colors_list[2], label="Interaction", alpha=0.88, edgecolor="#1a1a1a", linewidth=1.5),
            mpatches.Patch(color=bar_colors_list[3], label="LUC", alpha=0.88, edgecolor="#1a1a1a", linewidth=1.5),
        ]
        ax_var.legend(
            handles=legend_patches,
            loc="upper right",
            frameon=True,
            framealpha=0.98,
            fontsize=24,
            edgecolor="#cccccc",
            fancybox=True,
            shadow=True,
        )

    ax_var.text(
        0.02,
        0.97,
        "*** p<0.001 | ** p<0.01 | * p<0.05 | n.s.",
        transform=ax_var.transAxes,
        ha="left",
        va="top",
        fontsize=24,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffeb", alpha=0.92, edgecolor="#cccccc", linewidth=1.2),
        color="#1a1a1a",
        fontweight="bold",
    )

    ax_var.spines["top"].set_visible(False)
    ax_var.spines["right"].set_visible(False)
    ax_var.spines["left"].set_linewidth(2.5)
    ax_var.spines["bottom"].set_linewidth(2.5)
    ax_var.tick_params(axis="both", labelsize=26, width=2, length=5)


# Plot up-to net-zero (row 0)
plot_violin_and_bar_aff(
    fig,
    0,
    0,
    df_violin,
    unscaled_values,
    variance_df,
    universal_var,
    "Up-to Net-Zero",
    "Variance Decomposition (Factorial ANOVA: Up-to Net-Zero)",
    0,
    show_legend=False,
    violin_ylim=[min_aff_violin_y, max_aff_violin_y],
    variance_ylim=max_aff_variance_y,
)

# Plot post net-zero (row 1)
plot_violin_and_bar_aff(
    fig,
    1,
    0,
    df_violin_post_nz,
    unscaled_values_post_nz,
    variance_df_post_nz,
    universal_var_post_nz,
    "Post-Net-Zero",
    "Variance Decomposition (Factorial ANOVA: Post-Net-Zero)",
    2,
    show_legend=True,
    violin_ylim=[min_aff_violin_y, max_aff_violin_y],
    variance_ylim=max_aff_variance_y,
)

fig.suptitle("", fontsize=28, fontweight="bold", y=0.98, x=0.5, ha="center")

plt.savefig(OUTPUT_FIG_PNG, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.savefig(OUTPUT_FIG_PDF, bbox_inches="tight", facecolor="white", edgecolor="none")

print("\n" + "=" * 80)
print("AFFORESTATION ANALYSIS COMPLETED")
print("=" * 80)
print(f"""
Outputs:
- Figure PNG: {OUTPUT_FIG_PNG.name}
- Figure PDF: {OUTPUT_FIG_PDF.name}
- Stats XLSX: {OUTPUT_STATS_XLSX.name}
- LSMs included: {lsm_order}
""")

plt.show()