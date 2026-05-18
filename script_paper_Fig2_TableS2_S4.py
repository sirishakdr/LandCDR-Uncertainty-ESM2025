import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyam

# Optional: suppress pyam SyntaxWarning noise on Python 3.13
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"pyam\..*")

# --- Configuration ---
SCENARIO = "SSP1-26"
REGION = "World"
VAR_AFF_CO2 = "Carbon Sequestration|Land Use|Afforestation"  # MtCO2/yr
YEAR_START = 2020
YEAR_END = 2100
YEAR_MAX = 2100

# Set to tuple like (-3, 5) for fixed y-limits, else None for auto-global
Y_FIXED = None

NET_ZERO_YEARS = {
    "AIM/CGE 2.0": 2100,
    "MESSAGE-GLOBIOM 1.0": 2073,
    "GCAM 4.2": 2079,
    "IMAGE 3.0.1": 2076,
    "REMIND-MAGPIE 1.5": 2075,
}

# Publication-quality colors (no yellow)
CAT_COLORS = {
    "Unscaled": "#66c2a5",
    "nattoaff": "#fc8d62",
    "agtoaff": "#8da0cb",
    "combined": "#e78ac3",
}

# --- Input/output names ---
AR6_WORLD_NAME = "AR6_Scenarios_Database_World_v1.1.csv"
AR6_META_NAME = "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
AR6_META_SHEET = "meta_Ch3vetted_withclimate"
SCALING_FILE_NAME = "scaling_factors_afforestation_global_paper.xlsx"

OUT_TABLE_S2_NAME = "summary_global_overestimation_afforestastion_stats_paper_TableS2.xlsx"
OUT_TABLE_S4_NAME = "summary_global_afforestation_per_model_overestimation_stats_paper_TableS4.xlsx"
OUT_FIG_BASE_NAME = "paper_Fig2"


# --- Path resolution (portable, no local hardcoded path) ---
def resolve_data_dir(required_files):
    candidates = []

    # Optional override:
    # export PAPER_FILES_DIR=/path/to/paper_files
    env_dir = os.getenv("PAPER_FILES_DIR")
    if env_dir:
        p = Path(env_dir).expanduser().resolve()
        candidates.append(p)
        candidates.append(p / "paper_files")

    # Notebook / terminal cwd
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.append(cwd / "paper_files")

    # Script location
    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
        candidates.append(script_dir / "paper_files")
        candidates.append(script_dir.parent / "paper_files")

    # De-duplicate in order
    unique = []
    seen = set()
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


DATA_DIR = resolve_data_dir([AR6_WORLD_NAME, AR6_META_NAME, SCALING_FILE_NAME])

AR6_WORLD_FILE = DATA_DIR / AR6_WORLD_NAME
AR6_META_FILE = DATA_DIR / AR6_META_NAME
SF_FILE = DATA_DIR / SCALING_FILE_NAME

OUT_TABLE_S2 = DATA_DIR / OUT_TABLE_S2_NAME
OUT_TABLE_S4 = DATA_DIR / OUT_TABLE_S4_NAME
OUT_FIG_PNG = DATA_DIR / f"{OUT_FIG_BASE_NAME}.png"
OUT_FIG_PDF = DATA_DIR / f"{OUT_FIG_BASE_NAME}.pdf"


# --- Helpers ---
def detect_col(df, candidates, required=True, label="column"):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"Could not find {label}. Tried: {candidates}. Available: {list(df.columns)}")
    return None


def normalize_transition(v):
    s = str(v).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if s in {"nattoaff", "naturaltoafforestation", "naturaltoaff", "nat2aff"}:
        return "nattoaff"
    if s in {"agtoaff", "agriculturaltoafforestation", "agriculturaltoaff", "ag2aff"}:
        return "agtoaff"
    if s in {"agtonat", "agriculturaltonatural", "agtonatural"}:
        return "agtonat"
    return s


def _read_csv_with_checks(csv_path: Path) -> pd.DataFrame:
    probe = pd.read_csv(csv_path, nrows=3)
    if probe.shape[1] == 1:
        first_col = str(probe.columns[0]).strip().lower()
        first_val = str(probe.iloc[0, 0]).strip().lower() if len(probe) > 0 else ""
        if "git-lfs.github.com/spec/v1" in first_col or "git-lfs.github.com/spec/v1" in first_val:
            raise ValueError(
                "AR6 CSV appears to be a Git LFS pointer, not real data. "
                "Download actual AR6 CSV or run 'git lfs pull'."
            )

    df = pd.read_csv(csv_path, low_memory=False)
    # strip BOM / whitespace
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def _normalize_iamc_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {str(c).strip().lower(): c for c in df.columns}
    required = ["model", "scenario", "region", "variable", "unit"]

    rename = {}
    for target in required:
        if target in col_map:
            rename[col_map[target]] = target
        else:
            raise ValueError(
                f"Missing required IAMC column '{target}'. Available columns: {list(df.columns)}"
            )

    return df.rename(columns=rename)


def load_iam_world_dataframe(csv_path: Path) -> pyam.IamDataFrame:
    raw = _read_csv_with_checks(csv_path)
    raw = _normalize_iamc_columns(raw)
    return pyam.IamDataFrame(data=raw)


# --- Load scaling factors ---
sf_df = pd.read_excel(SF_FILE)
sf_df.columns = [c.strip() for c in sf_df.columns]

iam_col = detect_col(sf_df, ["IAM_model", "model"], required=True, label="IAM model column")
landmodel_col = detect_col(sf_df, ["LandModel", "Model"], required=False, label="land model column")
transition_col = detect_col(sf_df, ["Transition", "Landuse"], required=True, label="transition/landuse column")
scaling_col = detect_col(sf_df, ["scaling_factor", "Scaling_factor", "scalingFactor"], required=True, label="scaling factor column")

sf_df["IAM_model"] = sf_df[iam_col].astype(str).str.strip().str.lower()
sf_df["LandModel"] = (
    sf_df[landmodel_col].astype(str).str.strip().str.lower() if landmodel_col else "unknown"
)
sf_df["Transition"] = sf_df[transition_col].astype(str).str.strip()
sf_df["Transition_norm"] = sf_df["Transition"].apply(normalize_transition)
sf_df["scaling_factor"] = pd.to_numeric(sf_df[scaling_col], errors="coerce")


print(f"Original number of rows: {len(sf_df)}")
sf_df = sf_df[sf_df["LandModel"] != "jules"].copy()
print(f"After filtering JULES: {len(sf_df)}")
print(f"Remaining land models: {sorted(sf_df['LandModel'].unique())}")


print(f"Before filtering AgToNat: {len(sf_df)}")
sf_df = sf_df[sf_df["Transition_norm"] != "agtonat"].copy()
print(f"After filtering AgToNat: {len(sf_df)}")

sf_df = sf_df[np.isfinite(sf_df["scaling_factor"])].copy()
print(f"\nLoaded scaling factors for {len(sf_df)} rows (JULES + AgToNat excluded)\n")


# --- Load & compute unscaled (decadal) ---
def compute_unscaled_aff_df():
    print("Loading AR6 database...")
    data = load_iam_world_dataframe(AR6_WORLD_FILE)

    meta = pd.read_excel(AR6_META_FILE, sheet_name=AR6_META_SHEET)
    meta = meta.rename(columns={"Model": "model", "Scenario": "scenario"})
    data.set_meta(meta=meta.set_index(["model", "scenario"]))

    if SCENARIO not in set(data.scenario):
        raise ValueError(f"Scenario '{SCENARIO}' not found.")

    filtered = data.filter(scenario=SCENARIO, region=REGION)
    var = filtered.filter(variable=VAR_AFF_CO2, year=range(YEAR_START, YEAR_END + 1))

    if var.data.empty:
        raise ValueError(f"No data found for variable '{VAR_AFF_CO2}' and scenario '{SCENARIO}'.")

    decadal_years = sorted({int(y) for y in var.data["year"].unique() if int(y) % 10 == 0})
    var = var.filter(year=decadal_years)

    aff_df = var.timeseries().reset_index()
    year_cols = [
        c for c in aff_df.columns
        if isinstance(c, (int, np.integer)) or (isinstance(c, str) and c.isdigit())
    ]
    aff_df[year_cols] = aff_df[year_cols].apply(pd.to_numeric, errors="coerce")

    print(f"Downloaded {len(aff_df)} model-scenario pairs\n")
    return aff_df.set_index(["model", "scenario", "region", "variable", "unit"])


Afforestation_total_df = compute_unscaled_aff_df()


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


years_all = get_years_from_columns(Afforestation_total_df, YEAR_END)


def cumulative_mt(series: pd.Series, first_year: int, last_year: int) -> float:
    # pyam cumulative with non-overlapping split support
    if int(first_year) > int(last_year):
        return 0.0
    val = pyam.timeseries.cumulative(series, int(first_year), int(last_year))
    if val is None or not np.isfinite(val):
        return np.nan
    return float(val)


def _init_period_store():
    return {
        "unscaled": [],
        "nattoaff": [],
        "agtoaff": [],
        "combined": [],
        "unscaled_for_nattoaff": [],
        "unscaled_for_agtoaff": [],
        "unscaled_for_combined": [],
    }


def build_cumulative_distributions_per_model():
    """
    For each model in SSP1-26:
    - Up-to: [YEAR_START, NZ]
    - Post  : [NZ+1, YEAR_END]
    Uses pyam.timeseries.cumulative for integration.
    """
    model_names = []
    upto = _init_period_store()
    post = _init_period_store()
    split_checks = []

    models_unique = sorted(Afforestation_total_df.index.get_level_values("model").unique())
    total_models = 0
    skipped = 0

    print(f"Processing {len(models_unique)} models for {SCENARIO}...\n")

    for model in models_unique:
        model_lc = model.strip().lower()

        nz_year = int(NET_ZERO_YEARS.get(model, YEAR_END))
        if nz_year > YEAR_END or nz_year < YEAR_START:
            skipped += 1
            continue

        try:
            sel = Afforestation_total_df.loc[(model, SCENARIO, REGION, VAR_AFF_CO2)]
            row = sel.iloc[0] if isinstance(sel, pd.DataFrame) else sel
            sequestration_series = pd.Series(
                [row.get(y, row.get(str(y), np.nan)) for y in years_all],
                index=years_all
            )
        except KeyError:
            skipped += 1
            continue

        # Non-overlapping split to avoid NZ double counting
        cum_mt_upto = cumulative_mt(sequestration_series, YEAR_START, nz_year)
        cum_mt_post = cumulative_mt(sequestration_series, nz_year + 1, YEAR_END)
        cum_mt_full = cumulative_mt(sequestration_series, YEAR_START, YEAR_END)

        cum_gt_upto = cum_mt_upto / 1000.0 if np.isfinite(cum_mt_upto) else np.nan
        cum_gt_post = cum_mt_post / 1000.0 if np.isfinite(cum_mt_post) else np.nan
        cum_gt_full = cum_mt_full / 1000.0 if np.isfinite(cum_mt_full) else np.nan

        if not (np.isfinite(cum_gt_upto) and np.isfinite(cum_gt_post)):
            skipped += 1
            continue

        split_checks.append({
            "Model": model,
            "Full_2020_2100_GtCO2": cum_gt_full,
            "UpToNZ_GtCO2": cum_gt_upto,
            "PostNZ_GtCO2": cum_gt_post,
            "SplitSum_GtCO2": cum_gt_upto + cum_gt_post,
            "Delta_GtCO2": (cum_gt_upto + cum_gt_post - cum_gt_full) if np.isfinite(cum_gt_full) else np.nan,
        })

        total_models += 1
        model_names.append(model)

        sf_rows = sf_df[sf_df["IAM_model"] == model_lc]

        sf_nattoaff = sf_rows[sf_rows["Transition_norm"] == "nattoaff"]["scaling_factor"].values
        sf_nattoaff = sf_nattoaff[np.isfinite(sf_nattoaff)]

        sf_agtoaff = sf_rows[sf_rows["Transition_norm"] == "agtoaff"]["scaling_factor"].values
        sf_agtoaff = sf_agtoaff[np.isfinite(sf_agtoaff)]

        sf_combined = np.concatenate([sf_nattoaff, sf_agtoaff])

        for period_store, cum_gt in [(upto, cum_gt_upto), (post, cum_gt_post)]:
            period_store["unscaled"].append(cum_gt)

            nattoaff_vals = []
            for sf in sf_nattoaff:
                scaled_val = cum_gt * sf
                if np.isfinite(scaled_val):
                    nattoaff_vals.append(scaled_val)
                    period_store["unscaled_for_nattoaff"].append(cum_gt)
            period_store["nattoaff"].append(nattoaff_vals)

            agtoaff_vals = []
            for sf in sf_agtoaff:
                scaled_val = cum_gt * sf
                if np.isfinite(scaled_val):
                    agtoaff_vals.append(scaled_val)
                    period_store["unscaled_for_agtoaff"].append(cum_gt)
            period_store["agtoaff"].append(agtoaff_vals)

            combined_vals = []
            for sf in sf_combined:
                scaled_val = cum_gt * sf
                if np.isfinite(scaled_val):
                    combined_vals.append(scaled_val)
                    period_store["unscaled_for_combined"].append(cum_gt)
            period_store["combined"].append(combined_vals)

    print(f"Processed {total_models} models")
    print(f"Skipped {skipped} models with missing/invalid data")
    if model_names:
        print(f"Models: {', '.join(model_names)}\n")

    print("Up-to-Net-Zero Period:")
    print(f"  Unscaled values: {len(upto['unscaled'])}")
    print(f"  NatToAff per-model lists: {len(upto['nattoaff'])}")
    print(f"  AgToAff per-model lists: {len(upto['agtoaff'])}")
    print(f"  Combined per-model lists: {len(upto['combined'])}")

    print("\nPost-Net-Zero Period:")
    print(f"  Unscaled values: {len(post['unscaled'])}")
    print(f"  NatToAff per-model lists: {len(post['nattoaff'])}")
    print(f"  AgToAff per-model lists: {len(post['agtoaff'])}")
    print(f"  Combined per-model lists: {len(post['combined'])}\n")

    split_df = pd.DataFrame(split_checks)
    if not split_df.empty:
        print("Split consistency (UpTo + Post - Full), GtCO2:")
        print(split_df[["Model", "Delta_GtCO2"]].to_string(index=False))
        print()

    return {
        "model_names": model_names,
        "upto": upto,
        "post": post,
        "n_models": total_models,
        "split_check": split_df,
    }


def calculate_and_export_statistics_excel(all_data):
    aggregate_columns = [
        "Period",
        "LUC",
        "Unscaled Median (GtCO2)",
        "Unscaled Q25 (GtCO2)",
        "Unscaled Q75 (GtCO2)",
        "Scaled Median (GtCO2)",
        "Scaled Q25 (GtCO2)",
        "Scaled Q75 (GtCO2)",
        "Overestimation Median (times)",
        "Overestimation Q25 (times)",
        "Overestimation Q75 (times)",
    ]

    per_model_columns = [
        "Period",
        "Model",
        "LUC",
        "Unscaled Q25 (GtCO2)",
        "Unscaled Median (GtCO2)",
        "Unscaled Q75 (GtCO2)",
        "Scaled Q25 (GtCO2)",
        "Scaled Median (GtCO2)",
        "Scaled Q75 (GtCO2)",
        "Overestimation Q25 (times)",
        "Overestimation Median (times)",
        "Overestimation Q75 (times)",
    ]

    transition_labels = [
        ("nattoaff", "Natural -> Afforestation"),
        ("agtoaff", "Agricultural -> Afforestation"),
        ("combined", "Combined LUC (NatToAff + AgToAff)"),
    ]

    # ---------- Aggregate table (Table S2) ----------
    agg_rows = []

    for period_name, period_data in [("Up-to-Net-Zero", all_data["upto"]), ("Post-Net-Zero", all_data["post"])]:
        unscaled_clean = np.array([v for v in period_data["unscaled"] if np.isfinite(v)], dtype=float)
        if unscaled_clean.size == 0:
            continue

        u_med = float(np.median(unscaled_clean))
        u_q25 = float(np.percentile(unscaled_clean, 25))
        u_q75 = float(np.percentile(unscaled_clean, 75))

        for key, label in transition_labels:
            scaled_vals = np.array(
                [v for model_list in period_data[key] for v in model_list if np.isfinite(v)],
                dtype=float
            )
            unscaled_ref = np.array(
                [v for v in period_data[f"unscaled_for_{key}"] if np.isfinite(v)],
                dtype=float
            )

            if scaled_vals.size == 0 or unscaled_ref.size == 0:
                continue

            ratios = np.array(
                [u / abs(s) for u, s in zip(unscaled_ref, scaled_vals) if np.isfinite(s) and s != 0 and np.isfinite(u)],
                dtype=float
            )
            if ratios.size == 0:
                continue

            agg_rows.append({
                "Period": period_name,
                "LUC": label,
                "Unscaled Median (GtCO2)": u_med,
                "Unscaled Q25 (GtCO2)": u_q25,
                "Unscaled Q75 (GtCO2)": u_q75,
                "Scaled Median (GtCO2)": float(np.median(scaled_vals)),
                "Scaled Q25 (GtCO2)": float(np.percentile(scaled_vals, 25)),
                "Scaled Q75 (GtCO2)": float(np.percentile(scaled_vals, 75)),
                "Overestimation Median (times)": float(np.median(ratios)),
                "Overestimation Q25 (times)": float(np.percentile(ratios, 25)),
                "Overestimation Q75 (times)": float(np.percentile(ratios, 75)),
            })

    aggregate_df = pd.DataFrame(agg_rows, columns=aggregate_columns)
    with pd.ExcelWriter(OUT_TABLE_S2, engine="openpyxl") as writer:
        aggregate_df.to_excel(writer, sheet_name="TableS2", index=False)
        if "split_check" in all_data and isinstance(all_data["split_check"], pd.DataFrame) and not all_data["split_check"].empty:
            all_data["split_check"].to_excel(writer, sheet_name="SplitCheck", index=False)

    print(f"Exported aggregate statistics to {OUT_TABLE_S2.name}\n")

    # ---------- Per-model table (Table S4) ----------
    def calc_stats(vals, unscaled):
        vals_arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
        if vals_arr.size == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        s_q25 = float(np.percentile(vals_arr, 25))
        s_med = float(np.median(vals_arr))
        s_q75 = float(np.percentile(vals_arr, 75))

        if not np.isfinite(unscaled):
            return s_q25, s_med, s_q75, np.nan, np.nan, np.nan

        ratios = np.array([unscaled / abs(v) for v in vals_arr if v != 0], dtype=float)
        if ratios.size == 0:
            return s_q25, s_med, s_q75, np.nan, np.nan, np.nan

        r_q25 = float(np.percentile(ratios, 25))
        r_med = float(np.median(ratios))
        r_q75 = float(np.percentile(ratios, 75))
        return s_q25, s_med, s_q75, r_q25, r_med, r_q75

    per_model_rows = []

    for model_idx, model_name in enumerate(all_data["model_names"]):
        for period_name, period_data in [("Up-to-Net-Zero", all_data["upto"]), ("Post-Net-Zero", all_data["post"])]:
            unscaled_val = period_data["unscaled"][model_idx]
            u_q25 = unscaled_val if np.isfinite(unscaled_val) else np.nan
            u_med = unscaled_val if np.isfinite(unscaled_val) else np.nan
            u_q75 = unscaled_val if np.isfinite(unscaled_val) else np.nan

            # Unscaled row
            per_model_rows.append({
                "Period": period_name,
                "Model": model_name,
                "LUC": "Unscaled",
                "Unscaled Q25 (GtCO2)": u_q25,
                "Unscaled Median (GtCO2)": u_med,
                "Unscaled Q75 (GtCO2)": u_q75,
                "Scaled Q25 (GtCO2)": np.nan,
                "Scaled Median (GtCO2)": np.nan,
                "Scaled Q75 (GtCO2)": np.nan,
                "Overestimation Q25 (times)": np.nan,
                "Overestimation Median (times)": np.nan,
                "Overestimation Q75 (times)": np.nan,
            })

            # NatToAff row
            nattoaff_vals = period_data["nattoaff"][model_idx]
            n_s_q25, n_s_med, n_s_q75, n_r_q25, n_r_med, n_r_q75 = calc_stats(nattoaff_vals, unscaled_val)
            per_model_rows.append({
                "Period": period_name,
                "Model": model_name,
                "LUC": "Natural -> Afforestation",
                "Unscaled Q25 (GtCO2)": u_q25,
                "Unscaled Median (GtCO2)": u_med,
                "Unscaled Q75 (GtCO2)": u_q75,
                "Scaled Q25 (GtCO2)": n_s_q25,
                "Scaled Median (GtCO2)": n_s_med,
                "Scaled Q75 (GtCO2)": n_s_q75,
                "Overestimation Q25 (times)": n_r_q25,
                "Overestimation Median (times)": n_r_med,
                "Overestimation Q75 (times)": n_r_q75,
            })

            # AgToAff row
            agtoaff_vals = period_data["agtoaff"][model_idx]
            a_s_q25, a_s_med, a_s_q75, a_r_q25, a_r_med, a_r_q75 = calc_stats(agtoaff_vals, unscaled_val)
            per_model_rows.append({
                "Period": period_name,
                "Model": model_name,
                "LUC": "Agricultural -> Afforestation",
                "Unscaled Q25 (GtCO2)": u_q25,
                "Unscaled Median (GtCO2)": u_med,
                "Unscaled Q75 (GtCO2)": u_q75,
                "Scaled Q25 (GtCO2)": a_s_q25,
                "Scaled Median (GtCO2)": a_s_med,
                "Scaled Q75 (GtCO2)": a_s_q75,
                "Overestimation Q25 (times)": a_r_q25,
                "Overestimation Median (times)": a_r_med,
                "Overestimation Q75 (times)": a_r_q75,
            })

            # Combined row
            combined_vals = period_data["combined"][model_idx]
            c_s_q25, c_s_med, c_s_q75, c_r_q25, c_r_med, c_r_q75 = calc_stats(combined_vals, unscaled_val)
            per_model_rows.append({
                "Period": period_name,
                "Model": model_name,
                "LUC": "Combined LUC",
                "Unscaled Q25 (GtCO2)": u_q25,
                "Unscaled Median (GtCO2)": u_med,
                "Unscaled Q75 (GtCO2)": u_q75,
                "Scaled Q25 (GtCO2)": c_s_q25,
                "Scaled Median (GtCO2)": c_s_med,
                "Scaled Q75 (GtCO2)": c_s_q75,
                "Overestimation Q25 (times)": c_r_q25,
                "Overestimation Median (times)": c_r_med,
                "Overestimation Q75 (times)": c_r_q75,
            })

    per_model_df = pd.DataFrame(per_model_rows, columns=per_model_columns)
    with pd.ExcelWriter(OUT_TABLE_S4, engine="openpyxl") as writer:
        per_model_df.to_excel(writer, sheet_name="TableS4", index=False)

    print(f"Exported per-model statistics to {OUT_TABLE_S4.name}\n")

    # Console summary
    print("=" * 140)
    print(f"STATISTICAL SUMMARY - AFFORESTATION {SCENARIO} (Gt CO2, JULES + AgToNat EXCLUDED)")
    print("=" * 140)
    for _, row in aggregate_df.iterrows():
        print(f"\n{row['Period']} | {row['LUC']}")
        print(
            f"  Unscaled: Median={row['Unscaled Median (GtCO2)']:>7.2f}  "
            f"Q25={row['Unscaled Q25 (GtCO2)']:>7.2f}  Q75={row['Unscaled Q75 (GtCO2)']:>7.2f}"
        )
        print(
            f"  Scaled:   Median={row['Scaled Median (GtCO2)']:>7.2f}  "
            f"Q25={row['Scaled Q25 (GtCO2)']:>7.2f}  Q75={row['Scaled Q75 (GtCO2)']:>7.2f}"
        )
        print(
            f"  Overest.: Median={row['Overestimation Median (times)']:>7.2f}x  "
            f"Q25={row['Overestimation Q25 (times)']:>7.2f}x  Q75={row['Overestimation Q75 (times)']:>7.2f}x"
        )
    print("=" * 140 + "\n")


def style_bp(bp, color):
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("none")
        patch.set_linewidth(1.2)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(2.0)
        med.set_linestyle("-")
    for w in bp["whiskers"]:
        w.set_color("black")
        w.set_linewidth(1.0)
        w.set_linestyle("-")
    for c in bp["caps"]:
        c.set_color("black")
        c.set_linewidth(1.0)


def _safe_boxplot(ax, data_lists, positions, color):
    pairs = []
    for d, p in zip(data_lists, positions):
        arr = np.array([v for v in d if np.isfinite(v)], dtype=float)
        if arr.size > 0:
            pairs.append((arr, p))

    if not pairs:
        return

    data_clean = [d for d, _ in pairs]
    pos_clean = [p for _, p in pairs]
    bp = ax.boxplot(
        data_clean,
        positions=pos_clean,
        widths=0.12,
        patch_artist=True,
        manage_ticks=False,
        showfliers=False,
        whis=[5, 95],
    )
    style_bp(bp, color)


def plot_per_model_multiplot(all_data):
    models_list = all_data["model_names"]
    n_models = len(models_list)

    if n_models == 0:
        print("No model data available for plotting.")
        return

    all_vals_combined = []
    for period_data in [all_data["upto"], all_data["post"]]:
        for v in period_data["unscaled"]:
            if np.isfinite(v):
                all_vals_combined.append(v)
        for lst in [period_data["nattoaff"], period_data["agtoaff"], period_data["combined"]]:
            for vals in lst:
                for v in vals:
                    if np.isfinite(v):
                        all_vals_combined.append(v)

    if Y_FIXED is not None and len(Y_FIXED) == 2:
        y_limits = tuple(Y_FIXED)
    elif all_vals_combined:
        vmin = min(all_vals_combined)
        vmax = max(all_vals_combined)
        if np.isclose(vmin, vmax):
            pad = max(0.5, abs(vmax) * 0.05)
            y_limits = (vmin - pad, vmax + pad)
        else:
            rng = vmax - vmin
            pad = 0.08 * rng
            y_limits = (vmin - pad, vmax + pad)
    else:
        y_limits = (0, 1)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.major.width"] = 1.0
    plt.rcParams["ytick.major.width"] = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax_idx, (period_name, period_data) in enumerate(
        [("Up-to-Net-Zero", all_data["upto"]), ("Post-Net-Zero", all_data["post"])]
    ):
        ax = axes[ax_idx]

        x_centers = np.arange(n_models, dtype=float)
        group_width = 0.8
        offset = group_width / 4.0

        positions_unscaled = x_centers - 1.5 * offset
        positions_nattoaff = x_centers - 0.5 * offset
        positions_agtoaff = x_centers + 0.5 * offset
        positions_combined = x_centers + 1.5 * offset

        _safe_boxplot(ax, period_data["nattoaff"], positions_nattoaff, CAT_COLORS["nattoaff"])
        _safe_boxplot(ax, period_data["agtoaff"], positions_agtoaff, CAT_COLORS["agtoaff"])
        _safe_boxplot(ax, period_data["combined"], positions_combined, CAT_COLORS["combined"])

        unscaled_vals = period_data["unscaled"]
        for i, val in enumerate(unscaled_vals):
            if np.isfinite(val):
                ax.scatter(
                    positions_unscaled[i],
                    val,
                    marker="D",
                    s=120,
                    color=CAT_COLORS["Unscaled"],
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=4,
                    alpha=0.9,
                )

        ax.set_xlim(-1, n_models)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(models_list, rotation=15, fontsize=16, ha="center")
        ax.set_ylabel("Cumulative Afforestation Carbon (Gt CO$_2$)", fontsize=16, fontweight="bold")
        ax.set_title(period_name, fontsize=16, fontweight="bold", pad=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5, color="#cccccc")
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(1.0)

        ax.tick_params(axis="both", labelsize=16, colors="black", width=1.0, length=4)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(y_limits)

    fig.suptitle("", fontsize=14, fontweight="bold", y=0.98)

    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker="D",
            color="w",
            markerfacecolor=CAT_COLORS["Unscaled"],
            markeredgecolor="black",
            markersize=10,
            label="Unscaled",
            linewidth=0,
        ),
        mpatches.Patch(facecolor=CAT_COLORS["nattoaff"], edgecolor="none", alpha=0.8, linewidth=1.2, label="Natural -> Afforestation"),
        mpatches.Patch(facecolor=CAT_COLORS["agtoaff"], edgecolor="none", alpha=0.8, linewidth=1.2, label="Agricultural -> Afforestation"),
        mpatches.Patch(facecolor=CAT_COLORS["combined"], edgecolor="none", alpha=0.8, linewidth=1.2, label="Combined LUC"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        fontsize=14,
        ncol=1,
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor="none",
        framealpha=1.0,
        bbox_to_anchor=(0.85, 0.6),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(OUT_FIG_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(OUT_FIG_PDF, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT_FIG_PNG.name}")
    print(f"Saved: {OUT_FIG_PDF.name}\n")
    plt.show()


# --- Main Execution ---
print("=" * 70)
print("BUILDING CUMULATIVE DATA - UP-TO AND POST NET-ZERO")
print("=" * 70)
all_data = build_cumulative_distributions_per_model()

print("=" * 70)
print("CALCULATING AND EXPORTING STATISTICS")
print("=" * 70)
calculate_and_export_statistics_excel(all_data)

print("=" * 70)
print("CREATING PER-MODEL MULTIPLOT")
print("=" * 70)
plot_per_model_multiplot(all_data)

print("\nAll operations completed successfully.")
print(f"Saved tables: {OUT_TABLE_S2.name}, {OUT_TABLE_S4.name}")
print(f"Saved figure: {OUT_FIG_PNG.name}, {OUT_FIG_PDF.name}")