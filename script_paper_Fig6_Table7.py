#!/usr/bin/env python3
"""
Multi-region TOTAL BIOMASS CARBON cumulative analysis with up-to and post net-zero periods.
R10 REGIONAL ANALYSIS: ALL C1 SCENARIOS from AR6 database
Maps AR6 regions to R10 regional categories.
2-panel VERTICAL subplot visualization with combined statistics table (Excel).
Per-region and global statistics with All LUC combined.
Uses region-specific scaling factors and individual net-zero years.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyam
from typing import List, Dict, Tuple
from pathlib import Path

# --- Configuration ---
CATEGORY = "C1"
VAR_AG_CROPS = "Agricultural Production|Energy|Crops"  # million tDM/yr
CARBON_FRACTION = 0.485   # tC per tDM
CO2_TO_C = 44.0 / 12.0    # tCO2 per tC
VAR_CO2_EMISSIONS = "Emissions|CO2"

YEAR_START = 2020
YEAR_END = 2100
DECADAL_YEARS = [y for y in range(YEAR_START, YEAR_END + 1) if y % 10 == 0]

META_SHEET = "meta_Ch3vetted_withclimate"

AR6_R10_FILE_NAME = "AR6_Scenarios_Database_R10_regions_v1.1.csv"
AR6_WORLD_CSV_NAME = "AR6_Scenarios_Database_World_v1.1.csv"
META_FILE_NAME = "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
SCALING_FACTORS_FILE_NAME = "scaling_factors_regions_biomass_ssp126D_paper.xlsx"

OUTPUT_FIG_BASE_NAME = "paper_Fig6"
OUTPUT_TABLE_NAME = "paper_tableS7.xlsx"

REGION_MAPPING = {
    "R10AFRICA": [
        "Countries of Sub-Saharan Africa",
        "Algeria", "Egypt", "Libya", "Madagascar", "Morocco", "Nigeria",
        "Tunisia", "South Africa", "Kenya", "Ethiopia", "Angola"
    ],
    "R10CHINA": [
        "China",
        "Countries of centrally-planned Asia; primarily China"
    ],
    "R10EUROPE": [
        "Eastern and Western Europe (i.e., the EU28)",
        "European Union (28 member countries)",
        "United Kingdom"
    ],
    "R10INDIA": [
        "Countries of South Asia; primarily India",
        "India"
    ],
    "R10LATIN_AM": [
        "Countries of Latin America and the Caribbean",
        "Latin American countries",
        "Brazil", "Mexico", "Argentina", "Colombia", "Chile", "Venezuela"
    ],
    "R10MIDDLE_EAST": [
        "Countries of the Middle East and Africa",
        "Countries of the Middle East; Iran, Iraq, Israel, Saudi Arabia, Qatar, etc.",
        "Saudi Arabia", "Turkey"
    ],
    "R10NORTH_AM": [
        "North America; primarily the United States of America and Canada",
        "United States of America", "Canada"
    ],
    "R10PAC_OECD": [
        "Pacific OECD",
        "Australia", "Japan", "South Korea", "Taiwan"
    ],
    "R10REF_ECON": [
        "Countries from the Reforming Economies of the Former Soviet Union",
        "Countries from the Reforming Ecomonies of the Former Soviet Union (R6)",
        "Reforming Economies of Eastern Europe and the Former Soviet Union; primarily Russia",
        "Russia"
    ],
    "R10REST_ASIA": [
        "Asian countries except Japan",
        "Asian countries except Japan (R6)",
        "Other countries of Asia",
        "Indonesia"
    ]
}

R10_REGIONS = list(REGION_MAPPING.keys())

CAT_COLORS = {
    "Unscaled": "#66c2a5",
    "agtobio": "#fc8d62",
    "nattobio": "#8da0cb",
    "all": "#ffd92f",
}

# --- DATA_DIR resolver (portable + GitHub-safe) ---
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

    # Script directory (when run as .py)
    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
        candidates.append(script_dir / "paper_files")
        candidates.append(script_dir.parent / "paper_files")

    # De-duplicate while preserving order
    unique_candidates = []
    seen = set()
    for c in candidates:
        k = str(c)
        if k not in seen:
            seen.add(k)
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
    AR6_R10_FILE_NAME,
    AR6_WORLD_CSV_NAME,
    META_FILE_NAME,
    SCALING_FACTORS_FILE_NAME,
]

DATA_DIR = resolve_data_dir(required_inputs)
DATA_DIR.mkdir(parents=True, exist_ok=True)

AR6_R10_FILE = DATA_DIR / AR6_R10_FILE_NAME
AR6_WORLD_CSV = DATA_DIR / AR6_WORLD_CSV_NAME
META_FILE = DATA_DIR / META_FILE_NAME
SCALING_FACTORS_FILE = DATA_DIR / SCALING_FACTORS_FILE_NAME

OUTPUT_FIG_BASE = DATA_DIR / OUTPUT_FIG_BASE_NAME
OUTPUT_TABLE = DATA_DIR / OUTPUT_TABLE_NAME


# --- Helper Functions ---

def make_region_key(region_name: str) -> str:
    key = str(region_name).strip().lower()
    if key.startswith("r10"):
        key = key[3:]
    elif key.startswith("r5"):
        key = key[2:]
    key = key.replace("-", "").replace("_", "").replace(" ", "")
    key = key.rstrip("+")
    return key


def detect_col(df: pd.DataFrame, candidates: List[str], required=True) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"None of required columns {candidates} found. Available: {list(df.columns)}")
    return ""


def load_scaling_factors(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Scaling factors file not found: {path.name}")
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    landuse_col = detect_col(df, ["Landuse", "landuse"])
    scaling_col = detect_col(df, ["scaling_factor", "Scaling_factor", "scalingFactor", "CarbonDealing_factor"])
    region_col = detect_col(df, ["Region", "Region_mapped", "region", "region_mapped", "R10", "r10"])
    for c in [landuse_col, region_col]:
        if c:
            df[c] = df[c].astype(str).str.strip()
    df["__region_key"] = df[region_col].apply(make_region_key)
    df["Landuse"] = df[landuse_col].str.lower()
    df["scaling_factor"] = pd.to_numeric(df[scaling_col], errors="coerce")
    df = df[df["scaling_factor"].notna() & np.isfinite(df["scaling_factor"])]
    print(f"[✓] Loaded {len(df)} scaling factor entries")
    return df[["__region_key", "Landuse", "scaling_factor"]]


def find_net_zero(years: np.ndarray, emissions: np.ndarray) -> float | None:
    for i in range(1, len(years)):
        if emissions[i - 1] > 0 and emissions[i] <= 0:
            t0, t1 = years[i - 1], years[i]
            e0, e1 = emissions[i - 1], emissions[i]
            if e0 - e1 != 0:
                frac = e0 / (e0 - e1)
                return t0 + frac * (t1 - t0)
            return float(t0)
    return None


def map_region_to_r10(region: str) -> str | None:
    region_norm = region.rstrip("+")
    if region_norm in R10_REGIONS:
        return region_norm
    for r10_region, ar6_regions in REGION_MAPPING.items():
        if region in ar6_regions or region_norm in ar6_regions:
            return r10_region
    return None


def cumulative_mt(series: pd.Series, first_year: int, last_year: int) -> float:
    val = pyam.timeseries.cumulative(series, first_year, last_year)
    if val is None or not np.isfinite(val):
        return np.nan
    return float(val)


# --- Main Data Loading ---

def compute_ag_and_netzero_data_r10():
    print("\n[*] Loading AR6 R10 regional CSV...")
    iam_r10 = pyam.IamDataFrame(data=str(AR6_R10_FILE))

    print("[*] Loading AR6 World CSV (for CO2 net-zero)...")
    iam_world = pyam.IamDataFrame(data=str(AR6_WORLD_CSV))

    print("[*] Loading metadata and filtering to C1...")
    meta = pd.read_excel(META_FILE, sheet_name=META_SHEET)
    meta = meta.rename(columns={"Model": "model", "Scenario": "scenario"})
    if "Category" not in meta.columns:
        raise ValueError("Metadata sheet is missing Category column.")
    meta["Category"] = meta["Category"].astype(str).str.strip()

    # C1-consistent filtering via metadata column
    meta_idx = meta.set_index(["model", "scenario"])
    iam_r10.set_meta(meta=meta_idx)
    iam_world.set_meta(meta=meta_idx)

    c1_pairs = meta[meta["Category"] == CATEGORY][["model", "scenario"]].drop_duplicates()
    print(f"    C1 pairs in metadata: {len(c1_pairs)}")

    print("[*] Loading Agricultural Energy Crops (decadal)...")
    ag_raw = iam_r10.filter(
        Category=CATEGORY,
        variable=VAR_AG_CROPS,
        year=DECADAL_YEARS
    )
    if ag_raw.data.empty:
        raise ValueError(f"No regional data found for {VAR_AG_CROPS}")

    # Strict runtime check: C1 only
    ag_meta = ag_raw.meta.reset_index()
    if not ag_meta["Category"].astype(str).str.strip().eq(CATEGORY).all():
        raise ValueError("Non-C1 scenarios detected in regional agricultural crops selection.")

    ag_ts = ag_raw.timeseries()

    year_cols = sorted([c for c in ag_ts.columns
                        if isinstance(c, (int, np.integer))
                        and YEAR_START <= c <= YEAR_END
                        and c % 10 == 0])

    # Convert million tDM/yr -> million tCO2/yr
    ag_ts = ag_ts * CARBON_FRACTION * CO2_TO_C

    print(f"    Decadal year columns: {year_cols}")
    print(f"    Agricultural crops rows: {len(ag_ts)}")

    # Net-zero from World CO2 with full-year resolution
    print("[*] Computing net-zero years from World CO2 (full year resolution)...")
    co2_raw = iam_world.filter(
        Category=CATEGORY,
        variable=VAR_CO2_EMISSIONS,
        region="World"
    )
    if co2_raw.data.empty:
        raise ValueError("No World CO2 emissions data found for C1 scenarios.")

    co2_meta = co2_raw.meta.reset_index()
    if not co2_meta["Category"].astype(str).str.strip().eq(CATEGORY).all():
        raise ValueError("Non-C1 scenarios detected in World CO2 selection.")

    co2_ts = co2_raw.timeseries()

    net_zero_dict = {}
    co2_year_cols = sorted([c for c in co2_ts.columns
                            if isinstance(c, (int, np.integer))
                            and YEAR_START <= c <= YEAR_END])
    co2_idx = co2_ts.index
    for i in range(len(co2_ts)):
        model = co2_idx.get_level_values("model")[i]
        scenario = co2_idx.get_level_values("scenario")[i]
        emissions = co2_ts.iloc[i][co2_year_cols].values.astype(float) / 1000.0
        nz = find_net_zero(np.array(co2_year_cols), emissions)
        if nz is not None and YEAR_START <= nz <= YEAR_END:
            net_zero_dict[(model, scenario)] = nz

    print(f"[✓] Net-zero years found for {len(net_zero_dict)} scenarios\n")
    return ag_ts, net_zero_dict, year_cols


# --- Cumulative distribution gathering ---

def gather_cumulative_distributions_per_region(
    ag_ts: pd.DataFrame,
    year_cols: List[int],
    scaling_factors_df: pd.DataFrame,
    net_zero_dict: Dict[Tuple[str, str], float],
    period_type: str
) -> Dict[str, Dict[str, List[float]]]:
    """
    period_type: upto or post
    Always returns all 10 R10 regions.
    """
    result: Dict[str, Dict[str, List[float]]] = {
        r10: {
            "unscaled": [],
            "agtobio": [],
            "nattobio": [],
            "unscaled_for_agtobio": [],
            "unscaled_for_nattobio": [],
        }
        for r10 in R10_REGIONS
    }

    region_scaling = {}
    for _, row in scaling_factors_df.iterrows():
        rk = row["__region_key"]
        lu = row["Landuse"]
        sf = row["scaling_factor"]
        if rk not in region_scaling:
            region_scaling[rk] = {"agtobio": [], "nattobio": []}
        if lu in ("agtobio", "nattobio"):
            region_scaling[rk][lu].append(sf)

    print(f"[✓] Built scaling factor lookup for {len(region_scaling)} region keys")
    print(f"[DEBUG] Region keys in scaling factors: {sorted(region_scaling.keys())}")

    valid_ms = set(net_zero_dict.keys())
    idx = ag_ts.index
    mask = [(m, s) in valid_ms for m, s in zip(idx.get_level_values("model"), idx.get_level_values("scenario"))]
    filtered = ag_ts[mask]
    print(f"[*] Rows after net-zero filter: {len(filtered)} / {len(ag_ts)}")

    print("\n[DEBUG] R10 region key -> scaling factor availability:")
    for r10 in R10_REGIONS:
        rk = make_region_key(r10)
        sc = region_scaling.get(rk, {})
        print(f"  {r10} -> key='{rk}' agtobio={len(sc.get('agtobio', []))} nattobio={len(sc.get('nattobio', []))}")
    print()

    ar6_regions_in_data = sorted(set(filtered.index.get_level_values("region")))
    scenarios_processed = 0
    scenarios_with_data = 0

    for ar6_region in ar6_regions_in_data:
        r10_region = map_region_to_r10(ar6_region)
        if r10_region is None:
            continue

        region_key = make_region_key(r10_region)
        scales = region_scaling.get(region_key, {"agtobio": [], "nattobio": []})

        try:
            region_df = filtered.xs(ar6_region, level="region")
        except KeyError:
            continue

        region_idx = region_df.index
        models = region_idx.get_level_values("model")
        scenarios = region_idx.get_level_values("scenario")

        for i in range(len(region_df)):
            model = models[i]
            scenario = scenarios[i]
            scenarios_processed += 1

            nz_year = net_zero_dict.get((model, scenario))
            if nz_year is None or not np.isfinite(nz_year):
                continue
            nz_int = int(np.round(nz_year))
            if not (YEAR_START <= nz_int <= YEAR_END):
                continue

            row = region_df.iloc[i]
            series = pd.Series(row[year_cols].values.astype(float), index=year_cols)

            # Non-overlapping split to remove double counting
            if period_type == "upto":
                start_year = YEAR_START
                end_year = nz_int
            else:
                start_year = nz_int + 1
                end_year = YEAR_END

            if start_year > end_year:
                continue

            cum_mt = cumulative_mt(series, start_year, end_year)
            if not np.isfinite(cum_mt):
                continue

            cum_gt = cum_mt / 1000.0
            scenarios_with_data += 1
            result[r10_region]["unscaled"].append(cum_gt)

            for sf in scales["agtobio"]:
                if np.isfinite(sf):
                    sv = cum_gt * sf
                    if np.isfinite(sv):
                        result[r10_region]["agtobio"].append(sv)
                        result[r10_region]["unscaled_for_agtobio"].append(cum_gt)

            for sf in scales["nattobio"]:
                if np.isfinite(sf):
                    sv = cum_gt * sf
                    if np.isfinite(sv):
                        result[r10_region]["nattobio"].append(sv)
                        result[r10_region]["unscaled_for_nattobio"].append(cum_gt)

    print(f"[✓] Processed {scenarios_processed} scenario-region combos, {scenarios_with_data} with valid data")
    return result


# --- Statistics export (Excel) ---

def calculate_and_export_statistics(
    result_upto: Dict[str, Dict[str, List[float]]],
    result_post: Dict[str, Dict[str, List[float]]],
    output_file: Path
) -> pd.DataFrame:
    results = []

    luc_labels = [
        ("agtobio", "Agricultural to Bioenergy"),
        ("nattobio", "Natural land to Bioenergy"),
    ]

    for period_name, result_dict in [("Up-to Net-Zero", result_upto),
                                     ("Post Net-Zero", result_post)]:
        regions = sorted(result_dict.keys())

        for region in regions:
            d = result_dict[region]
            unscaled_arr = np.array([v for v in d.get("unscaled", []) if np.isfinite(v)])
            agtobio_arr = np.array([v for v in d.get("agtobio", []) if np.isfinite(v)])
            nattobio_arr = np.array([v for v in d.get("nattobio", []) if np.isfinite(v)])
            unscaled_for_agtobio = np.array([v for v in d.get("unscaled_for_agtobio", []) if np.isfinite(v)])
            unscaled_for_nattobio = np.array([v for v in d.get("unscaled_for_nattobio", []) if np.isfinite(v)])

            u_med = np.median(unscaled_arr) if len(unscaled_arr) > 0 else np.nan
            u_q25 = np.percentile(unscaled_arr, 25) if len(unscaled_arr) > 0 else np.nan
            u_q75 = np.percentile(unscaled_arr, 75) if len(unscaled_arr) > 0 else np.nan

            for key, label in luc_labels:
                scaled_arr = agtobio_arr if key == "agtobio" else nattobio_arr
                unscaled_ref = unscaled_for_agtobio if key == "agtobio" else unscaled_for_nattobio

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

            # All LUC Combined per region
            all_scaled = np.concatenate([agtobio_arr, nattobio_arr])
            all_unscref = np.concatenate([unscaled_for_agtobio, unscaled_for_nattobio])

            if len(all_scaled) > 0:
                a_med = np.median(all_scaled)
                a_q25 = np.percentile(all_scaled, 25)
                a_q75 = np.percentile(all_scaled, 75)
                ratios = np.array(
                    [u / abs(s) for u, s in zip(all_unscref, all_scaled)
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

        for key, label in luc_labels:
            scaled_arr = g_agtobio if key == "agtobio" else g_nattobio
            unscaled_ref = g_uref_ag if key == "agtobio" else g_uref_nat

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

    results_df = pd.DataFrame(results)

    # Save as Excel table S7
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="TableS7", index=False)

    print(f"[✓] Statistics exported to {output_file.name}")
    return results_df


# --- Plotting ---

def style_bp(bp, color):
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("none")
        patch.set_linewidth(0)
    for med in bp["medians"]:
        med.set_color("#333333")
        med.set_linewidth(1.8)
    for w in bp["whiskers"]:
        w.set_color("#555555")
    for c in bp["caps"]:
        c.set_color("#555555")


def plot_periods_subplots_vertical(
    result_upto: Dict[str, Dict[str, List[float]]],
    result_post: Dict[str, Dict[str, List[float]]],
    outfile_base: Path
):
    fig, axes = plt.subplots(2, 1, figsize=(20, 14))
    period_configs = [
        (0, "Up-to Net-Zero", result_upto),
        (1, "Post Net-Zero", result_post),
    ]

    all_regions = R10_REGIONS

    for ax_idx, period_label, result_dict in period_configs:
        ax = axes[ax_idx]
        n_regions = len(all_regions)
        x_centers = np.arange(n_regions)
        group_width = 0.8
        offset = group_width / 5.0
        positions = {
            t: x_centers - 1.5 * offset + i * offset
            for i, t in enumerate(["unscaled", "agtobio", "nattobio", "all"])
        }

        def safe_boxplot(data_lists, pos_arr, color):
            pairs = [(d, p) for d, p in zip(data_lists, pos_arr)
                     if len(d) > 0 and any(np.isfinite(v) for v in d)]
            if not pairs:
                return
            d_clean, p_clean = zip(*pairs)
            bp = ax.boxplot(list(d_clean), positions=list(p_clean), widths=0.15,
                            patch_artist=True, manage_ticks=False, showfliers=False)
            style_bp(bp, color)

        safe_boxplot([result_dict[r]["unscaled"] for r in all_regions], positions["unscaled"], CAT_COLORS["Unscaled"])
        safe_boxplot([result_dict[r]["agtobio"] for r in all_regions], positions["agtobio"], CAT_COLORS["agtobio"])
        safe_boxplot([result_dict[r]["nattobio"] for r in all_regions], positions["nattobio"], CAT_COLORS["nattobio"])

        data_all, pos_all = [], []
        for i, r in enumerate(all_regions):
            combined = [v for v in result_dict[r]["agtobio"] + result_dict[r]["nattobio"] if np.isfinite(v)]
            if combined:
                data_all.append(combined)
                pos_all.append(positions["all"][i])
        if data_all:
            bp_all = ax.boxplot(data_all, positions=pos_all, widths=0.15,
                                patch_artist=True, manage_ticks=False, showfliers=False)
            style_bp(bp_all, CAT_COLORS["all"])

        ax.set_xticks(x_centers)
        ax.set_xticklabels([r.replace("R10", "") for r in all_regions], rotation=30, ha="center", fontsize=16, fontweight="bold")
        ax.set_xlim(-1, n_regions)
        ax.set_ylabel("Cumulative Bioenergy Carbon (GtCO$_2$)", fontsize=16, fontweight="bold")
        ax.set_title(period_label, fontsize=13, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_ylim(0, 400)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    legend_elems = [
        mpatches.Patch(facecolor=CAT_COLORS["Unscaled"], edgecolor="none", label="Unscaled"),
        mpatches.Patch(facecolor=CAT_COLORS["agtobio"], edgecolor="none", label="Agric -> Bioenergy"),
        mpatches.Patch(facecolor=CAT_COLORS["nattobio"], edgecolor="none", label="Natural -> Bioenergy"),
        mpatches.Patch(facecolor=CAT_COLORS["all"], edgecolor="none", label="All LUC Combined"),
    ]
    fig.legend(handles=legend_elems, loc="upper left", ncol=1,
               bbox_to_anchor=(0.8, 0.9), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(str(outfile_base) + ".png", dpi=600, bbox_inches="tight")
    plt.savefig(str(outfile_base) + ".pdf", bbox_inches="tight")
    print(f"[✓] Saved {outfile_base.name}.png and {outfile_base.name}.pdf")
    plt.show()


# --- Main ---

def main():
    print("=" * 80)
    print("R10 REGIONAL AGRICULTURAL ENERGY CROPS ANALYSIS — C1 SCENARIOS")
    print("=" * 80)

    print("\n[*] Loading region-specific scaling factors...")
    scaling_factors_df = load_scaling_factors(SCALING_FACTORS_FILE)

    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    ag_ts, net_zero_dict, year_cols = compute_ag_and_netzero_data_r10()

    nz_years = [y for y in net_zero_dict.values() if np.isfinite(y)]
    if nz_years:
        print(
            f"Net-zero stats: mean={np.mean(nz_years):.1f}  median={np.median(nz_years):.1f}"
            f"  min={np.min(nz_years):.1f}  max={np.max(nz_years):.1f}  N={len(nz_years)}\n"
        )

    print("=" * 80)
    print("UP-TO NET-ZERO ANALYSIS")
    print("=" * 80)
    upto_dist = gather_cumulative_distributions_per_region(
        ag_ts, year_cols, scaling_factors_df, net_zero_dict, "upto"
    )

    print("\n[*] Up-to net-zero data summary (all 10 regions):")
    for r in R10_REGIONS:
        d = upto_dist[r]
        note = " <- NO DATA" if not d["unscaled"] else ""
        print(
            f"  {r:20s}: unscaled={len(d['unscaled'])}"
            f"  agtobio={len(d['agtobio'])}  nattobio={len(d['nattobio'])}{note}"
        )

    print("\n" + "=" * 80)
    print("POST NET-ZERO ANALYSIS")
    print("=" * 80)
    post_dist = gather_cumulative_distributions_per_region(
        ag_ts, year_cols, scaling_factors_df, net_zero_dict, "post"
    )

    print("\n[*] Post net-zero data summary (all 10 regions):")
    for r in R10_REGIONS:
        d = post_dist[r]
        note = " <- NO DATA" if not d["unscaled"] else ""
        print(
            f"  {r:20s}: unscaled={len(d['unscaled'])}"
            f"  agtobio={len(d['agtobio'])}  nattobio={len(d['nattobio'])}{note}"
        )

    print("\n" + "=" * 80)
    print("CREATING 2-PANEL VERTICAL SUBPLOT")
    print("=" * 80)
    plot_periods_subplots_vertical(
        upto_dist,
        post_dist,
        outfile_base=OUTPUT_FIG_BASE
    )

    print("\n" + "=" * 80)
    print("EXPORTING STATISTICS TABLE (EXCEL)")
    print("=" * 80)
    stats_df = calculate_and_export_statistics(upto_dist, post_dist, OUTPUT_TABLE)
    print(stats_df.head(20).to_string(index=False))

    print("\n" + "=" * 80)
    print("[✓] ALL ANALYSES COMPLETED")
    print("=" * 80)
    print(f"Saved figure: {OUTPUT_FIG_BASE_NAME}.png/.pdf")
    print(f"Saved table:  {OUTPUT_TABLE_NAME}")


if __name__ == "__main__":
    main()