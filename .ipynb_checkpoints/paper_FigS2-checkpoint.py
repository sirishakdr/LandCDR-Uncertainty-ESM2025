#!/usr/bin/env python3
"""
Multi-region TOTAL BIOMASS CARBON cumulative analysis with up-to and post net-zero periods for SSP126D.
2-panel subplot visualization with combined statistics table (Excel).
Per-region and global statistics with All LUC combined.

Consistent with pyam-based integration and no split-year double counting:
- pyam.timeseries.cumulative() for cumulative integration
- model-specific fixed net-zero years
-non-overlapping split:
  Up-to = [YEAR_START, nz_year]
  Post  = [nz_year + 1, YEAR_MAX]

"""

from __future__ import annotations
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyam
from typing import List, Dict, Tuple
from pathlib import Path

# --- Configuration ---
SCENARIO = "SSP1_SPA1_26I_D"
SCALING_FALLBACK_SCENARIO = "SSP1-26"
ALLOW_SCENARIO_BASE_MATCH = True
KEEP_ALL_SCALING_IF_NO_MATCH = True

VAR_AG_CROPS = "Agricultural Production|Energy|Crops"  # million tDM/yr
CARBON_FRACTION = 0.485        # tC per tDM
CO2_TO_C = 44.0 / 12.0         # tCO2 per tC

YEAR_START = 2020
YEAR_END = 2100
YEAR_MAX = 2100
DECADAL_ONLY = True

PREFERRED_MODELS = [
    "AIM/CGE 2.0",
    "GCAM 4.2",
    "IMAGE 3.0.1",
    "IMAGE 3.2",
    "MESSAGE-GLOBIOM 1.0",
    "REMIND-MAGPIE 1.5",
]

NET_ZERO_YEARS = {
    "AIM/CGE 2.0": 2100,
    "GCAM 4.2": 2079,
    "IMAGE 3.0.1": 2076,
    "IMAGE 3.2": 2072,
    "MESSAGE-GLOBIOM 1.0": 2073,
    "REMIND-MAGPIE 1.5": 2075,
}

CAT_COLORS = {
    "Unscaled": "#66c2a5",
    "agtobio": "#fc8d62",
    "nattobio": "#8da0cb",
    "all": "#ffd92f",
}

UNSCALED_JITTER_SD = 0.04
UNSCALED_POINT_SIZE = 38
UNSCALED_MEAN_SIZE = 78

HIDE_UNSCALED_MEAN_IF_SINGLE = True
ANNOTATE_REGION_MODEL_COUNTS = True

AUTO_DETECT_REGIONS = True
MANUAL_REGIONS: List[str] = []

AR6_DATA_CSV_NAME = "AR6_Scenarios_Database_R10_regions_v1.1.csv"
AR6_META_XLSX_NAME = "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
AR6_META_SHEET = "meta_Ch3vetted_withclimate"
SCALING_FACTORS_FILE_NAME = "scaling_factors_regions_biomass_ssp126D_paper.xlsx"

OUTPUT_FIG_BASE_NAME = "paper_FigS2"
OUTPUT_TABLE_NAME = "regional_ssp126D_biomass_stats_paper.xlsx"

RNG_SEED = 42


# --- DATA_DIR resolver (portable + no local hardcoded path) ---
def resolve_data_dir(required_files: List[str]) -> Path:
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


required_inputs = [
    AR6_DATA_CSV_NAME,
    AR6_META_XLSX_NAME,
    SCALING_FACTORS_FILE_NAME,
]

DATA_DIR = resolve_data_dir(required_inputs)
DATA_DIR.mkdir(parents=True, exist_ok=True)

AR6_DATA_CSV = DATA_DIR / AR6_DATA_CSV_NAME
AR6_META_XLSX = DATA_DIR / AR6_META_XLSX_NAME
SCALING_FACTORS_FILE = DATA_DIR / SCALING_FACTORS_FILE_NAME

OUTPUT_FIG_BASE = DATA_DIR / OUTPUT_FIG_BASE_NAME
OUTPUT_TABLE = DATA_DIR / OUTPUT_TABLE_NAME


# --- Helper Functions ---

def scenario_base(s: str) -> str:
    s = str(s).lower().strip()
    for ch in ["-", " "]:
        s = s.replace(ch, "_")
    while "__" in s:
        s = s.replace("__", "_")
    m = re.search(r"(ssp\d+)[_-]?(\d{2})", s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return s


def make_region_key(s: str) -> str:
    s = str(s).strip().lower()
    if s.startswith("r10"):
        s = s[3:]
    if s.startswith("r5"):
        s = s[2:]
    for ch in ["_", "-", " "]:
        s = s.replace(ch, "")
    return s


def detect_col(df: pd.DataFrame, candidates: List[str], required=True) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of required columns {candidates} found. Available: {list(df.columns)}")
    return ""


def resolve_scenario(data: pyam.IamDataFrame, desired: str, fallback_generic: str | None) -> str:
    scenarios = set(data.scenario)
    if desired in scenarios:
        return desired

    desired_base = scenario_base(desired)
    candidates = [s for s in scenarios if scenario_base(s) == desired_base]
    if candidates:
        print(f"[INFO] Scenario '{desired}' not found; using base-match '{candidates[0]}'")
        return candidates[0]

    if fallback_generic and fallback_generic in scenarios:
        print(f"[INFO] Using fallback scenario '{fallback_generic}'")
        return fallback_generic

    if fallback_generic:
        fb_base = scenario_base(fallback_generic)
        fb_candidates = [s for s in scenarios if scenario_base(s) == fb_base]
        if fb_candidates:
            print(f"[INFO] Using fallback base-match '{fb_candidates[0]}'")
            return fb_candidates[0]

    raise ValueError(f"No matching scenario for '{desired}' or fallback '{fallback_generic}'.")


def load_scaling_factors(path: Path, target_scenario: str, fallback_generic: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    landuse_col = detect_col(df, ["Landuse", "landuse"])
    scaling_col = detect_col(df, ["scaling_factor", "Scaling_factor", "scalingFactor", "scalingfactor"])
    region_col = detect_col(df, ["Region", "Region_mapped", "region", "region_mapped", "R10", "r10"])
    scenario_col = detect_col(df, ["SSP_std", "Scenario", "scenario"], required=False)

    for c in [landuse_col, region_col] + ([scenario_col] if scenario_col else []):
        if c:
            df[c] = df[c].astype(str).str.strip()

    df["__region_key"] = df[region_col].apply(make_region_key)
    df["Landuse"] = df[landuse_col].astype(str).str.lower()
    df["scaling_factor"] = pd.to_numeric(df[scaling_col], errors="coerce")

    if scenario_col:
        df["__scenario_base"] = df[scenario_col].apply(scenario_base)
        bases_avail = set(df["__scenario_base"])
        target_base = scenario_base(target_scenario)
        fallback_base = scenario_base(fallback_generic) if fallback_generic else None

        chosen = None
        if ALLOW_SCENARIO_BASE_MATCH:
            if target_base in bases_avail:
                chosen = target_base
            elif fallback_base and fallback_base in bases_avail:
                print(f"[INFO] Scaling factors base match using fallback '{fallback_base}'")
                chosen = fallback_base

        if chosen:
            before = len(df)
            df = df[df["__scenario_base"] == chosen]
            print(f"Filtered scaling factors to base '{chosen}': {len(df)} rows (from {before})")
        else:
            if not KEEP_ALL_SCALING_IF_NO_MATCH:
                print("[WARN] No scaling scenario/base match. Dropping all scaling rows.")
                df = df.iloc[0:0]
            else:
                print(f"[WARN] No scaling scenario/base match. Keeping all rows ({len(df)}).")
    else:
        print("[INFO] Scaling factors file has no scenario column; using all rows.")

    # Keep JULES rows in biomass workflow
    df = df[np.isfinite(df["scaling_factor"])].copy()
    return df[["__region_key", "Landuse", "scaling_factor"]]


def load_iam_data(csv_path: Path, meta_xlsx: Path, meta_sheet: str) -> pyam.IamDataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not meta_xlsx.exists():
        raise FileNotFoundError(meta_xlsx)

    data = pyam.IamDataFrame(data=str(csv_path))
    meta = pd.read_excel(meta_xlsx, sheet_name=meta_sheet)
    meta = meta.rename(columns={"Model": "model", "Scenario": "scenario"})
    data.set_meta(meta=meta.set_index(["model", "scenario"]))
    return data


def extract_decadal(filtered: pyam.IamDataFrame, variable: str) -> pd.DataFrame:
    var_df = filtered.filter(variable=variable, year=range(YEAR_START, YEAR_END + 1))
    years = sorted({int(y) for y in var_df.data["year"].unique()})
    if DECADAL_ONLY:
        years = [y for y in years if y % 10 == 0]
    var_df = var_df.filter(year=years)
    ts = var_df.timeseries().reset_index()

    year_cols = [
        c for c in ts.columns
        if isinstance(c, (int, np.integer)) or (isinstance(c, str) and c.isdigit())
    ]
    rename_map = {c: str(c) for c in year_cols}
    if rename_map:
        ts = ts.rename(columns=rename_map)

    ycols = [str(c) for c in year_cols]
    ts[ycols] = ts[ycols].apply(pd.to_numeric, errors="coerce")
    return ts


def compute_ag_carbon_df(data: pyam.IamDataFrame, scenario: str) -> pd.DataFrame:
    if scenario not in set(data.scenario):
        raise ValueError(f"Scenario '{scenario}' not found after resolution.")
    filtered = data.filter(scenario=scenario)

    if VAR_AG_CROPS not in set(filtered.variable):
        raise ValueError(f"Variable '{VAR_AG_CROPS}' missing for '{scenario}'.")

    ag = extract_decadal(filtered, VAR_AG_CROPS)
    ycols = [c for c in ag.columns if isinstance(c, str) and c.isdigit()]

    # million tDM/yr -> GtCO2/yr  (million tCO2/yr / 1000)
    ag[ycols] = ag[ycols] * CARBON_FRACTION * CO2_TO_C / 1000.0

    ag["variable"] = "Total Biomass Carbon (derived)"
    ag = ag.set_index(["model", "scenario", "region", "variable", "unit"])
    return ag


def available_models(total_df: pd.DataFrame) -> List[str]:
    return sorted(set(total_df.index.get_level_values("model")))


def choose_model_order(total_df: pd.DataFrame) -> List[str]:
    present = available_models(total_df)
    ordered = [m for m in PREFERRED_MODELS if m in present]
    if not ordered:
        print("[WARN] Preferred models absent; using all available.")
        return present
    extras = [m for m in present if m not in ordered]
    return ordered + extras


def extract_unscaled_series(total_df: pd.DataFrame, scenario: str, model: str, region: str) -> Tuple[np.ndarray, np.ndarray]:
    key = (model, scenario, region, "Total Biomass Carbon (derived)")
    try:
        sel = total_df.loc[key]
    except KeyError:
        return np.array([]), np.array([])

    row = sel if isinstance(sel, pd.Series) else sel.iloc[0]
    years = [int(c) for c in row.index if isinstance(c, str) and c.isdigit()]
    years_sorted = sorted(years)
    vals = row[[str(y) for y in years_sorted]].values.astype(float)
    return np.array(years_sorted, dtype=int), np.array(vals, dtype=float)


def cumulative_mt(series: pd.Series, first_year: int, last_year: int) -> float:
    val = pyam.timeseries.cumulative(series, int(first_year), int(last_year))
    if val is None or not np.isfinite(val):
        return np.nan
    return float(val)


def gather_cumulative_distributions(
    total_df: pd.DataFrame,
    scaling_df: pd.DataFrame,
    scenario: str,
    regions: List[str],
    model_order: List[str],
    period_type: str,
) -> Dict[str, Dict[str, List[float]]]:
    """period_type: 'upto' or 'post'."""
    result: Dict[str, Dict[str, List[float]]] = {}
    models_present = available_models(total_df)
    model_map = {m.lower(): m for m in models_present}

    for region in regions:
        if (total_df.index.get_level_values("region") == region).sum() == 0:
            continue

        region_key = make_region_key(region)
        scaling_region = (
            scaling_df[scaling_df["__region_key"] == region_key]
            if not scaling_df.empty
            else pd.DataFrame()
        )

        unscaled_vals: List[float] = []
        agtobio_vals: List[float] = []
        nattobio_vals: List[float] = []
        unscaled_for_agtobio: List[float] = []
        unscaled_for_nattobio: List[float] = []

        for label in model_order:
            m_actual = model_map.get(label.lower())
            if m_actual is None:
                continue

            yrs, vals = extract_unscaled_series(total_df, scenario, m_actual, region)
            if yrs.size == 0:
                continue

            series = pd.Series(vals, index=yrs)

            nz_year = int(NET_ZERO_YEARS.get(label, YEAR_MAX))
            if nz_year < YEAR_START or nz_year > YEAR_MAX:
                continue

            # Non-overlapping split (fixes double counting)
            if period_type == "upto":
                start_year = YEAR_START
                end_year = nz_year
            else:
                start_year = nz_year + 1
                end_year = YEAR_MAX

            if start_year > end_year:
                cum_val = 0.0
            else:
                cum_val = cumulative_mt(series, start_year, end_year)

            if not np.isfinite(cum_val):
                continue

            unscaled_vals.append(cum_val)

            if not scaling_region.empty:
                for sfv in scaling_region[scaling_region["Landuse"] == "agtobio"]["scaling_factor"]:
                    if np.isfinite(sfv):
                        agtobio_vals.append(cum_val * sfv)
                        unscaled_for_agtobio.append(cum_val)

                for sfv in scaling_region[scaling_region["Landuse"] == "nattobio"]["scaling_factor"]:
                    if np.isfinite(sfv):
                        nattobio_vals.append(cum_val * sfv)
                        unscaled_for_nattobio.append(cum_val)

        if unscaled_vals:
            result[region] = {
                "unscaled": unscaled_vals,
                "agtobio": agtobio_vals,
                "nattobio": nattobio_vals,
                "unscaled_for_agtobio": unscaled_for_agtobio,
                "unscaled_for_nattobio": unscaled_for_nattobio,
            }

    return result


def calculate_and_export_statistics(
    result_upto: Dict[str, Dict[str, List[float]]],
    result_post: Dict[str, Dict[str, List[float]]],
    output_file: Path,
) -> pd.DataFrame:
    results = []
    col_order = [
        "Period",
        "Region",
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

    luc_labels = [
        ("agtobio", "unscaled_for_agtobio", "Agricultural to Bioenergy"),
        ("nattobio", "unscaled_for_nattobio", "Natural land to Bioenergy"),
    ]

    for period_name, result_dict in [("Up-to Net-Zero", result_upto), ("Post Net-Zero", result_post)]:
        regions = sorted(result_dict.keys())

        # Per-region
        for region in regions:
            d = result_dict[region]
            unscaled_arr = np.array([v for v in d.get("unscaled", []) if np.isfinite(v)])
            agtobio_arr = np.array([v for v in d.get("agtobio", []) if np.isfinite(v)])
            nattobio_arr = np.array([v for v in d.get("nattobio", []) if np.isfinite(v)])
            uref_ag = np.array([v for v in d.get("unscaled_for_agtobio", []) if np.isfinite(v)])
            uref_nat = np.array([v for v in d.get("unscaled_for_nattobio", []) if np.isfinite(v)])

            u_med = np.median(unscaled_arr) if len(unscaled_arr) > 0 else np.nan
            u_q25 = np.percentile(unscaled_arr, 25) if len(unscaled_arr) > 0 else np.nan
            u_q75 = np.percentile(unscaled_arr, 75) if len(unscaled_arr) > 0 else np.nan

            for key, _, label in luc_labels:
                scaled_arr = agtobio_arr if key == "agtobio" else nattobio_arr
                unscaled_ref = uref_ag if key == "agtobio" else uref_nat

                if len(scaled_arr) > 0:
                    s_med = np.median(scaled_arr)
                    s_q25 = np.percentile(scaled_arr, 25)
                    s_q75 = np.percentile(scaled_arr, 75)
                    ratios = np.array(
                        [u / abs(s) for u, s in zip(unscaled_ref, scaled_arr)
                         if s != 0 and np.isfinite(u) and np.isfinite(s)]
                    )
                    r_med = np.median(ratios) if len(ratios) else np.nan
                    r_q25 = np.percentile(ratios, 25) if len(ratios) else np.nan
                    r_q75 = np.percentile(ratios, 75) if len(ratios) else np.nan
                else:
                    s_med = s_q25 = s_q75 = r_med = r_q25 = r_q75 = np.nan

                results.append({
                    "Period": period_name,
                    "Region": region,
                    "LUC": label,
                    "Unscaled Median (GtCO2)": u_med,
                    "Unscaled Q25 (GtCO2)": u_q25,
                    "Unscaled Q75 (GtCO2)": u_q75,
                    "Scaled Median (GtCO2)": s_med,
                    "Scaled Q25 (GtCO2)": s_q25,
                    "Scaled Q75 (GtCO2)": s_q75,
                    "Overestimation Median (times)": r_med,
                    "Overestimation Q25 (times)": r_q25,
                    "Overestimation Q75 (times)": r_q75,
                })

            # All LUC Combined per region (pooled values)
            all_scaled = np.concatenate([agtobio_arr, nattobio_arr])
            all_uref = np.concatenate([uref_ag, uref_nat])

            if len(all_scaled) > 0:
                a_med = np.median(all_scaled)
                a_q25 = np.percentile(all_scaled, 25)
                a_q75 = np.percentile(all_scaled, 75)
                ratios = np.array(
                    [u / abs(s) for u, s in zip(all_uref, all_scaled)
                     if s != 0 and np.isfinite(u) and np.isfinite(s)]
                )
                r_med = np.median(ratios) if len(ratios) else np.nan
                r_q25 = np.percentile(ratios, 25) if len(ratios) else np.nan
                r_q75 = np.percentile(ratios, 75) if len(ratios) else np.nan
            else:
                a_med = a_q25 = a_q75 = r_med = r_q25 = r_q75 = np.nan

            results.append({
                "Period": period_name,
                "Region": region,
                "LUC": "All LUC Combined",
                "Unscaled Median (GtCO2)": u_med,
                "Unscaled Q25 (GtCO2)": u_q25,
                "Unscaled Q75 (GtCO2)": u_q75,
                "Scaled Median (GtCO2)": a_med,
                "Scaled Q25 (GtCO2)": a_q25,
                "Scaled Q75 (GtCO2)": a_q75,
                "Overestimation Median (times)": r_med,
                "Overestimation Q25 (times)": r_q25,
                "Overestimation Q75 (times)": r_q75,
            })

        # Global aggregation
        g_unscaled = np.array([v for r in regions for v in result_dict[r].get("unscaled", []) if np.isfinite(v)])
        g_agtobio = np.array([v for r in regions for v in result_dict[r].get("agtobio", []) if np.isfinite(v)])
        g_nattobio = np.array([v for r in regions for v in result_dict[r].get("nattobio", []) if np.isfinite(v)])
        g_uref_ag = np.array([v for r in regions for v in result_dict[r].get("unscaled_for_agtobio", []) if np.isfinite(v)])
        g_uref_nat = np.array([v for r in regions for v in result_dict[r].get("unscaled_for_nattobio", []) if np.isfinite(v)])

        gu_med = np.median(g_unscaled) if len(g_unscaled) > 0 else np.nan
        gu_q25 = np.percentile(g_unscaled, 25) if len(g_unscaled) > 0 else np.nan
        gu_q75 = np.percentile(g_unscaled, 75) if len(g_unscaled) > 0 else np.nan

        for key, uref_arr, label in [("agtobio", g_uref_ag, "Agricultural to Bioenergy"),
                                     ("nattobio", g_uref_nat, "Natural land to Bioenergy")]:
            scaled_arr = g_agtobio if key == "agtobio" else g_nattobio
            if len(scaled_arr) > 0:
                s_med = np.median(scaled_arr)
                s_q25 = np.percentile(scaled_arr, 25)
                s_q75 = np.percentile(scaled_arr, 75)
                ratios = np.array(
                    [u / abs(s) for u, s in zip(uref_arr, scaled_arr)
                     if s != 0 and np.isfinite(u) and np.isfinite(s)]
                )
                r_med = np.median(ratios) if len(ratios) else np.nan
                r_q25 = np.percentile(ratios, 25) if len(ratios) else np.nan
                r_q75 = np.percentile(ratios, 75) if len(ratios) else np.nan
            else:
                s_med = s_q25 = s_q75 = r_med = r_q25 = r_q75 = np.nan

            results.append({
                "Period": period_name,
                "Region": "GLOBAL",
                "LUC": label,
                "Unscaled Median (GtCO2)": gu_med,
                "Unscaled Q25 (GtCO2)": gu_q25,
                "Unscaled Q75 (GtCO2)": gu_q75,
                "Scaled Median (GtCO2)": s_med,
                "Scaled Q25 (GtCO2)": s_q25,
                "Scaled Q75 (GtCO2)": s_q75,
                "Overestimation Median (times)": r_med,
                "Overestimation Q25 (times)": r_q25,
                "Overestimation Q75 (times)": r_q75,
            })

        g_all_scaled = np.concatenate([g_agtobio, g_nattobio])
        g_all_uref = np.concatenate([g_uref_ag, g_uref_nat])

        if len(g_all_scaled) > 0:
            a_med = np.median(g_all_scaled)
            a_q25 = np.percentile(g_all_scaled, 25)
            a_q75 = np.percentile(g_all_scaled, 75)
            ratios = np.array(
                [u / abs(s) for u, s in zip(g_all_uref, g_all_scaled)
                 if s != 0 and np.isfinite(u) and np.isfinite(s)]
            )
            r_med = np.median(ratios) if len(ratios) else np.nan
            r_q25 = np.percentile(ratios, 25) if len(ratios) else np.nan
            r_q75 = np.percentile(ratios, 75) if len(ratios) else np.nan
        else:
            a_med = a_q25 = a_q75 = r_med = r_q25 = r_q75 = np.nan

        results.append({
            "Period": period_name,
            "Region": "GLOBAL",
            "LUC": "All LUC Combined",
            "Unscaled Median (GtCO2)": gu_med,
            "Unscaled Q25 (GtCO2)": gu_q25,
            "Unscaled Q75 (GtCO2)": gu_q75,
            "Scaled Median (GtCO2)": a_med,
            "Scaled Q25 (GtCO2)": a_q25,
            "Scaled Q75 (GtCO2)": a_q75,
            "Overestimation Median (times)": r_med,
            "Overestimation Q25 (times)": r_q25,
            "Overestimation Q75 (times)": r_q75,
        })

    results_df = pd.DataFrame(results, columns=col_order)

    # Save as Excel table
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Stats", index=False)

    print(f"Statistics exported to {output_file.name}")
    return results_df


def style_bp(bp, color):
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for med in bp["medians"]:
        med.set_color("#333333")
        med.set_linewidth(1.8)
    for w in bp["whiskers"]:
        w.set_color("#555555")
    for c in bp["caps"]:
        c.set_color("#555555")


def plot_periods_subplots(
    result_upto: Dict[str, Dict[str, List[float]]],
    result_post: Dict[str, Dict[str, List[float]]],
    outfile_base: str,
):
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    fig.suptitle("", fontsize=16, fontweight="bold", y=1.02)

    period_configs = [
        (0, "Up-to Net-Zero", result_upto),
        (1, "Post Net-Zero", result_post),
    ]

    for ax_idx, period_label, result_dict in period_configs:
        ax = axes[ax_idx]
        regions = list(result_dict.keys())
        n_regions = len(regions)
        x_centers = np.arange(n_regions)
        group_width = 0.6
        offset = group_width / 4.0

        positions = {
            t: x_centers - 1.5 * offset + i * offset
            for i, t in enumerate(["agtobio", "nattobio", "all"])
        }

        data_agtobio = [result_dict[r]["agtobio"] for r in regions]
        data_nattobio = [result_dict[r]["nattobio"] for r in regions]
        data_un = [result_dict[r]["unscaled"] for r in regions]

        has_agtobio = any(np.isfinite(v) for lst in data_agtobio for v in lst)
        has_nattobio = any(np.isfinite(v) for lst in data_nattobio for v in lst)

        if has_agtobio:
            bp = ax.boxplot(
                data_agtobio,
                positions=positions["agtobio"],
                widths=0.15,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
            )
            style_bp(bp, CAT_COLORS["agtobio"])

        if has_nattobio:
            bp = ax.boxplot(
                data_nattobio,
                positions=positions["nattobio"],
                widths=0.15,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
            )
            style_bp(bp, CAT_COLORS["nattobio"])

        data_all = []
        for r in regions:
            combined = [v for v in result_dict[r]["agtobio"] + result_dict[r]["nattobio"] if np.isfinite(v)]
            data_all.append(combined if combined else [np.nan])

        bp = ax.boxplot(
            data_all,
            positions=positions["all"],
            widths=0.15,
            patch_artist=True,
            manage_ticks=False,
            showfliers=False,
        )
        style_bp(bp, CAT_COLORS["all"])

        rng = np.random.default_rng(RNG_SEED)
        for i, vals in enumerate(data_un):
            fin = [v for v in vals if np.isfinite(v)]
            if not fin:
                continue
            if len(fin) == 1:
                ax.scatter(
                    x_centers[i], fin[0],
                    marker="D", s=120,
                    color=CAT_COLORS["Unscaled"],
                    edgecolor="none", alpha=0.95, zorder=5
                )
            else:
                for v in fin:
                    jit = rng.normal(0, UNSCALED_JITTER_SD)
                    ax.scatter(
                        x_centers[i] + jit, v,
                        marker="D", s=120,
                        color=CAT_COLORS["Unscaled"],
                        edgecolor="none", alpha=0.85, zorder=4
                    )
                if not (HIDE_UNSCALED_MEAN_IF_SINGLE and len(fin) == 1):
                    ax.scatter(
                        x_centers[i], np.mean(fin),
                        marker="D", s=120,
                        color=CAT_COLORS["Unscaled"],
                        edgecolor="none", zorder=6
                    )

        ax.set_xticks(x_centers)
        ax.set_xticklabels(regions, rotation=30, ha="center", fontsize=16)
        ax.set_xlim(-1, n_regions)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_ylabel("Cumulative Bioenergy Carbon (GtCO$_2$)", fontsize=16, fontweight="bold")
        ax.set_title(period_label, fontsize=16, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 80)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    legend_elems = [
        mpatches.Patch(facecolor=CAT_COLORS["agtobio"], edgecolor="none", linewidth=1.5, label="Agric -> Bioenergy"),
        mpatches.Patch(facecolor=CAT_COLORS["nattobio"], edgecolor="none", linewidth=1.5, label="Natural -> Bioenergy"),
        mpatches.Patch(facecolor=CAT_COLORS["all"], edgecolor="none", linewidth=1.5, label="All LUC Combined"),
        plt.Line2D(
            [0], [0],
            marker="D", color="w",
            markerfacecolor=CAT_COLORS["Unscaled"],
            markeredgecolor="none",
            markersize=14,
            linestyle="None",
            label="Unscaled"
        ),
    ]
    fig.legend(handles=legend_elems, loc="upper right", ncol=1, bbox_to_anchor=(0.95, 0.90), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    plt.savefig(f"{outfile_base}.png", dpi=600, bbox_inches="tight")
    plt.savefig(f"{outfile_base}.pdf", bbox_inches="tight")
    print(f"Saved {Path(outfile_base).name}.png and {Path(outfile_base).name}.pdf")
    plt.show()


def main():
    print(f"Requested scenario variant: {SCENARIO}")
    data = load_iam_data(AR6_DATA_CSV, AR6_META_XLSX, AR6_META_SHEET)

    print("Resolving scenario variant...")
    resolved_scenario = resolve_scenario(data, SCENARIO, SCALING_FALLBACK_SCENARIO)
    print(f"Resolved IAM scenario: {resolved_scenario}")

    print("Loading scaling factors...")
    scaling_df = load_scaling_factors(
        SCALING_FACTORS_FILE,
        target_scenario=resolved_scenario,
        fallback_generic=SCALING_FALLBACK_SCENARIO,
    )
    print(f"Scaling factor rows retained: {len(scaling_df)}")

    print("Computing agricultural energy crops timeseries...")
    total_df = compute_ag_carbon_df(data, resolved_scenario)

    regions_all = sorted(set(total_df.index.get_level_values("region")))
    print(f"Regions present: {regions_all}")

    regions = regions_all if AUTO_DETECT_REGIONS else [r for r in MANUAL_REGIONS if r in regions_all]
    if not regions:
        raise RuntimeError("No regions selected/available.")

    model_order = choose_model_order(total_df)
    print("Model order used:", model_order)

    print("\n" + "=" * 80)
    print("UP-TO NET-ZERO ANALYSIS")
    print("=" * 80)
    upto_dist = gather_cumulative_distributions(
        total_df, scaling_df, resolved_scenario, regions, model_order, period_type="upto"
    )
    for r, d in upto_dist.items():
        print(f"[UP-TO NZ] {r}: unscaled={len(d['unscaled'])}, agtobio={len(d['agtobio'])}, nattobio={len(d['nattobio'])}")

    print("\n" + "=" * 80)
    print("POST NET-ZERO ANALYSIS")
    print("=" * 80)
    post_dist = gather_cumulative_distributions(
        total_df, scaling_df, resolved_scenario, regions, model_order, period_type="post"
    )
    for r, d in post_dist.items():
        print(f"[POST NZ] {r}: unscaled={len(d['unscaled'])}, agtobio={len(d['agtobio'])}, nattobio={len(d['nattobio'])}")

    print("\n" + "=" * 80)
    print("CREATING 2-PANEL SUBPLOT VISUALIZATION")
    print("=" * 80)
    plot_periods_subplots(
        upto_dist,
        post_dist,
        outfile_base=str(OUTPUT_FIG_BASE),
    )

    print("\n" + "=" * 80)
    print("CALCULATING COMBINED STATISTICS")
    print("=" * 80)
    stats_df = calculate_and_export_statistics(upto_dist, post_dist, OUTPUT_TABLE)
    print("\nCombined Statistics (Up-to and Post Net-Zero):")
    print(stats_df.to_string(index=False))

    print("\nSaved outputs:")
    print(f"  {OUTPUT_FIG_BASE_NAME}.png")
    print(f"  {OUTPUT_FIG_BASE_NAME}.pdf")
    print(f"  {OUTPUT_TABLE_NAME}")


if __name__ == "__main__":
    main()