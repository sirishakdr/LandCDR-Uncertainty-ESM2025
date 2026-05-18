#!/usr/bin/env python3
"""
Regional Afforestation C1 analysis
- Uses AR6 R10 regional CSV + metadata for explicit C1 filtering.
- Uses World CO2 to compute model-scenario-specific net-zero years.
- Uses pyam cumulative integration with non-overlapping windows:
  Up-to = [YEAR_START, NZ]
  Post  = [NZ+1, YEAR_END]

Outputs:
- Figure: paper_Fig7.png / paper_Fig7.pdf
- Stats : TableS8.xlsx
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyam

# --- Configuration ---
CATEGORY = "C1"
VAR_AFF = "Carbon Sequestration|Land Use|Afforestation"
VAR_CO2 = "Emissions|CO2"
YEAR_START = 2020
YEAR_END = 2100
DECADAL_YEARS = [y for y in range(YEAR_START, YEAR_END + 1) if y % 10 == 0]

# Input filenames
AR6_R10_FILE_NAME = "AR6_Scenarios_Database_R10_regions_v1.1.csv"
AR6_WORLD_FILE_NAME = "AR6_Scenarios_Database_World_v1.1.csv"
META_FILE_NAME = "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
META_SHEET = "meta_Ch3vetted_withclimate"
SCALING_FILE_NAME = "scaling_factors_regions_afforestation_paper.xlsx"

# Outputs requested
OUTPUT_FIG_BASE = "paper_Fig7"
OUTPUT_STATS_NAME = "TableS8.xlsx"

# AgToNat removed
TRANSITION_ORDER = ["nattoaff", "agtoaff"]

CAT_COLORS = {
    "Unscaled": "#66c2a5",
    "nattoaff": "#fc8d62",
    "agtoaff": "#8da0cb",
    "all": "#e78ac3",
}

# R10 regions — match normalized names (without trailing +)
R10_REGIONS = [
    "R10AFRICA",
    "R10CHINA",
    "R10EUROPE",
    "R10INDIA",
    "R10LATIN_AM",
    "R10MIDDLE_EAST",
    "R10NORTH_AM",
    "R10PAC_OECD",
    "R10REF_ECON",
    "R10REST_ASIA",
]


# --- Path resolution (portable, no local hardcoded paths) ---
def resolve_data_dir(required_files: List[str]) -> Path:
    candidates = []

    # Optional override:
    # export PAPER_FILES_DIR=/path/to/paper_files
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


DATA_DIR = resolve_data_dir(
    [AR6_R10_FILE_NAME, AR6_WORLD_FILE_NAME, META_FILE_NAME, SCALING_FILE_NAME]
)

AR6_R10_FILE = DATA_DIR / AR6_R10_FILE_NAME
AR6_WORLD_CSV = DATA_DIR / AR6_WORLD_FILE_NAME
META_FILE = DATA_DIR / META_FILE_NAME
SCALING_FILE = DATA_DIR / SCALING_FILE_NAME

OUTPUT_STATS_XLSX = DATA_DIR / OUTPUT_STATS_NAME
OUTPUT_FIG_PNG = DATA_DIR / f"{OUTPUT_FIG_BASE}.png"
OUTPUT_FIG_PDF = DATA_DIR / f"{OUTPUT_FIG_BASE}.pdf"


# --- Helpers ---
def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _pick_col(df: pd.DataFrame, aliases: List[str], required=True, label="column") -> Optional[str]:
    cmap = {_norm_col(c): c for c in df.columns}
    for a in aliases:
        k = _norm_col(a)
        if k in cmap:
            return cmap[k]
    if required:
        raise ValueError(f"Missing {label}. Tried aliases: {aliases}. Available: {list(df.columns)}")
    return None


def make_region_key(region_name: str) -> str:
    key = str(region_name).strip().lower()
    if key.startswith("r10"):
        key = key[3:]
    elif key.startswith("r5"):
        key = key[2:]
    key = key.replace("-", "").replace("_", "").replace(" ", "")
    key = key.rstrip("+")
    return key


def map_region_to_r10(region: str) -> Optional[str]:
    region_norm = str(region).strip().rstrip("+")
    if region_norm in R10_REGIONS:
        return region_norm
    return None


def find_net_zero(years: np.ndarray, emissions: np.ndarray) -> Optional[float]:
    for i in range(1, len(years)):
        if emissions[i - 1] > 0 and emissions[i] <= 0:
            t0, t1 = years[i - 1], years[i]
            e0, e1 = emissions[i - 1], emissions[i]
            if e0 - e1 != 0:
                return t0 + e0 / (e0 - e1) * (t1 - t0)
            return float(t0)
    return None


def cumulative_mt(series: pd.Series, first_year: int, last_year: int) -> float:
    # Supports non-overlapping post window when NZ == YEAR_END
    if int(first_year) > int(last_year):
        return 0.0
    val = pyam.timeseries.cumulative(series, int(first_year), int(last_year))
    if val is None or not np.isfinite(val):
        return np.nan
    return float(val)


def summarize(arr: np.ndarray) -> Tuple[float, float, float]:
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    return (
        float(np.median(arr)),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 75)),
    )


def normalize_transition(v: str) -> str:
    s = _norm_col(v)
    if s in {"nattoaff", "naturaltoafforestation", "naturaltoaff", "nat2aff"}:
        return "nattoaff"
    if s in {"agtoaff", "agriculturaltoafforestation", "agriculturaltoaff", "ag2aff"}:
        return "agtoaff"
    if s in {"agtonat", "agriculturaltonatural", "agtonatural"}:
        return "agtonat"
    if "aff" in s and "nat" in s and "agto" not in s:
        return "nattoaff"
    if "aff" in s and ("ag" in s or "agric" in s or "crop" in s):
        return "agtoaff"
    return s


# --- Load scaling factors ---
def load_scaling_factors() -> Dict[str, Dict[str, List[float]]]:
    sf = pd.read_excel(SCALING_FILE)
    sf.columns = [c.strip() for c in sf.columns]

    col_region = _pick_col(
        sf,
        ["region", "region_mapped", "r10", "r10region", "regionmapped"],
        required=True,
        label="region column",
    )
    col_trans = _pick_col(
        sf,
        ["transition", "landuse", "luc", "land_transition", "landtransition", "category"],
        required=True,
        label="transition column",
    )
    col_sf = _pick_col(
        sf,
        ["scaling_factor", "scalingfactor", "sf"],
        required=True,
        label="scaling factor column",
    )
    col_lsm = _pick_col(
        sf,
        ["lsm", "landmodel", "land_model", "model"],
        required=False,
        label="lsm column",
    )

    sf = sf.rename(columns={col_region: "region", col_trans: "transition", col_sf: "scaling_factor"})
    if col_lsm:
        sf = sf.rename(columns={col_lsm: "lsm"})

   
    print(f"Original rows: {len(sf)}")
    if "lsm" in sf.columns:
        sf["lsm"] = sf["lsm"].astype(str).str.strip().str.lower()
        sf = sf[sf["lsm"] != "jules"].copy()
        print(f"After filtering JULES: {len(sf)}")
        print(f"Remaining land models: {sorted(sf['lsm'].dropna().unique())}")
    else:
        print("[WARNING] 'lsm' column not found — cannot filter JULES")

    print("\n*** EXCLUDING AgToNat FROM ALL ANALYSES ***")
    print(f"Before AgToNat filter: {len(sf)}")
    sf["transition"] = sf["transition"].apply(normalize_transition)
    sf = sf[sf["transition"] != "agtonat"].copy()
    print(f"After filtering AgToNat: {len(sf)}")

    sf["__region_key"] = sf["region"].apply(make_region_key)
    sf["scaling_factor"] = pd.to_numeric(sf["scaling_factor"], errors="coerce")
    sf = sf[np.isfinite(sf["scaling_factor"])].copy()
    sf = sf[sf["transition"].isin(TRANSITION_ORDER)].copy()

    region_scaling: Dict[str, Dict[str, List[float]]] = {}
    for (rk, trans), grp in sf.groupby(["__region_key", "transition"]):
        if rk not in region_scaling:
            region_scaling[rk] = {t: [] for t in TRANSITION_ORDER}
        region_scaling[rk][trans] = grp["scaling_factor"].tolist()

    print("\n[DEBUG] Region keys in scaling factors:")
    print(sorted(region_scaling.keys()))
    return region_scaling


# --- Load data ---
def load_local_data():
    print("\n[*] Loading AR6 R10 regional CSV...")
    iam_r10 = pyam.IamDataFrame(data=str(AR6_R10_FILE))

    print("[*] Loading AR6 World CSV (CO2 net-zero)...")
    iam_world = pyam.IamDataFrame(data=str(AR6_WORLD_CSV))

    print("[*] Loading metadata and filtering to C1...")
    meta = pd.read_excel(META_FILE, sheet_name=META_SHEET)
    meta = meta.rename(columns={"Model": "model", "Scenario": "scenario"})
    if "Category" not in meta.columns:
        raise ValueError("Metadata sheet is missing 'Category' column.")
    meta["Category"] = meta["Category"].astype(str).str.strip()

    meta_idx = meta.set_index(["model", "scenario"])
    iam_r10.set_meta(meta=meta_idx)
    iam_world.set_meta(meta=meta_idx)

    c1_pairs = meta[meta["Category"] == CATEGORY][["model", "scenario"]].drop_duplicates()
    print(f"    C1 pairs in metadata: {len(c1_pairs)}")

    aff_filtered = iam_r10.filter(
        Category=CATEGORY,
        variable=VAR_AFF,
        year=DECADAL_YEARS
    )
    if aff_filtered.data.empty:
        raise ValueError("No afforestation data found for C1 in R10 CSV")
    aff_meta = aff_filtered.meta.reset_index()
    if not aff_meta["Category"].astype(str).str.strip().eq(CATEGORY).all():
        raise ValueError("Non-C1 scenarios detected in R10 afforestation selection.")
    aff_ts = aff_filtered.timeseries()

    co2_filtered = iam_world.filter(
        Category=CATEGORY,
        variable=VAR_CO2,
        region="World"
    )
    if co2_filtered.data.empty:
        raise ValueError("No CO2 emissions data found for C1")
    co2_meta = co2_filtered.meta.reset_index()
    if not co2_meta["Category"].astype(str).str.strip().eq(CATEGORY).all():
        raise ValueError("Non-C1 scenarios detected in World CO2 selection.")
    co2_ts = co2_filtered.timeseries()

    year_cols = sorted([
        c for c in aff_ts.columns
        if isinstance(c, (int, np.integer))
        and YEAR_START <= c <= YEAR_END
        and c % 10 == 0
    ])

    print(f"    Afforestation rows: {len(aff_ts)}  Decadal cols: {year_cols}")
    print(f"    CO2 rows: {len(co2_ts)} (all years)\n")
    return aff_ts, co2_ts, year_cols


def build_net_zero_dict(co2_ts: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    net_zero_dict: Dict[Tuple[str, str], float] = {}
    co2_year_cols = sorted([
        c for c in co2_ts.columns
        if isinstance(c, (int, np.integer))
        and YEAR_START <= c <= YEAR_END
    ])

    idx = co2_ts.index
    for i in range(len(co2_ts)):
        model = idx.get_level_values("model")[i]
        scenario = idx.get_level_values("scenario")[i]
        emissions = co2_ts.iloc[i][co2_year_cols].values.astype(float) / 1000.0
        nz = find_net_zero(np.array(co2_year_cols), emissions)
        if nz is not None and YEAR_START <= nz <= YEAR_END:
            net_zero_dict[(model, scenario)] = nz

    print(f"[✓] Net-zero years found for {len(net_zero_dict)} scenarios\n")
    return net_zero_dict


# --- Cumulative distributions ---
def gather_cumulative_distributions_per_region(
    aff_ts: pd.DataFrame,
    year_cols: List[int],
    region_scaling: Dict[str, Dict[str, List[float]]],
    net_zero_dict: Dict[Tuple[str, str], float],
    period_type: str
) -> Dict[str, Dict[str, List[float]]]:
    result: Dict[str, Dict[str, List[float]]] = {
        r10: {
            "unscaled": [],
            "nattoaff": [],
            "agtoaff": [],
            "unscaled_for_nattoaff": [],
            "unscaled_for_agtoaff": [],
        }
        for r10 in R10_REGIONS
    }

    valid_ms = set(net_zero_dict.keys())
    idx = aff_ts.index
    mask = [
        (m, s) in valid_ms
        for m, s in zip(idx.get_level_values("model"), idx.get_level_values("scenario"))
    ]
    filtered = aff_ts[mask]
    print(f"[*] Rows after net-zero filter: {len(filtered)} / {len(aff_ts)}")

    print("\n[DEBUG] R10 region key -> scaling factor availability:")
    for r10 in R10_REGIONS:
        rk = make_region_key(r10)
        sc = region_scaling.get(rk, {})
        print(
            f"  {r10} -> key='{rk}' "
            f"nattoaff={len(sc.get('nattoaff', []))} "
            f"agtoaff={len(sc.get('agtoaff', []))}"
        )
    print()

    ar6_regions_in_data = sorted(set(filtered.index.get_level_values("region")))
    processed = 0
    with_data = 0

    for ar6_region in ar6_regions_in_data:
        r10_region = map_region_to_r10(ar6_region)
        if r10_region is None:
            continue

        region_key = make_region_key(r10_region)
        scales = region_scaling.get(region_key, {t: [] for t in TRANSITION_ORDER})

        try:
            region_df = filtered.xs(ar6_region, level="region")
        except KeyError:
            continue

        ridx = region_df.index
        models = ridx.get_level_values("model")
        scenarios = ridx.get_level_values("scenario")

        for i in range(len(region_df)):
            model = models[i]
            scenario = scenarios[i]
            processed += 1

            nz_year = net_zero_dict.get((model, scenario))
            if nz_year is None or not np.isfinite(nz_year):
                continue
            nz_int = int(np.round(nz_year))
            if not (YEAR_START <= nz_int <= YEAR_END):
                continue

            row = region_df.iloc[i]
            series = pd.Series(row[year_cols].values.astype(float), index=year_cols)

            # Non-overlapping split
            if period_type == "upto":
                start_y, end_y = YEAR_START, nz_int
            else:
                start_y, end_y = nz_int + 1, YEAR_END

            cum_mt = cumulative_mt(series, start_y, end_y)
            if not np.isfinite(cum_mt):
                continue

            cum_gt = cum_mt / 1000.0
            with_data += 1
            result[r10_region]["unscaled"].append(cum_gt)

            for trans in TRANSITION_ORDER:
                for sf in scales.get(trans, []):
                    if np.isfinite(sf):
                        sv = cum_gt * sf
                        if np.isfinite(sv):
                            result[r10_region][trans].append(sv)
                            result[r10_region][f"unscaled_for_{trans}"].append(cum_gt)

    print(f"[✓] Processed {processed} scenario-region combos, {with_data} with valid data")
    return result


# --- Statistics (REGIONS ONLY, no GLOBAL rows) ---
def calculate_and_export_statistics(
    upto: Dict[str, Dict[str, List[float]]],
    post: Dict[str, Dict[str, List[float]]],
    output_file: Path
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
        "Overestimation/Underestimation Median (times)",
        "Overestimation/Underestimation Q25 (times)",
        "Overestimation/Underestimation Q75 (times)",
    ]

    luc_labels = {
        "nattoaff": "Natural to Afforestation",
        "agtoaff": "Agricultural to Afforestation",
        "all": "All (Combined)",
    }

    for period_name, result_dict in [("Up-to Net-Zero", upto), ("Post Net-Zero", post)]:
        for region in R10_REGIONS:
            d = result_dict[region]
            u_arr = np.array([v for v in d["unscaled"] if np.isfinite(v)], dtype=float)
            if len(u_arr) == 0:
                continue

            u_med, u_q25, u_q75 = summarize(u_arr)

            for trans in TRANSITION_ORDER:
                s_arr = np.array([v for v in d[trans] if np.isfinite(v)], dtype=float)
                ur_arr = np.array([v for v in d[f"unscaled_for_{trans}"] if np.isfinite(v)], dtype=float)

                s_med, s_q25, s_q75 = summarize(s_arr)
                ratios = np.array(
                    [u / abs(s) for u, s in zip(ur_arr, s_arr) if np.isfinite(u) and np.isfinite(s) and s != 0],
                    dtype=float
                )
                r_med, r_q25, r_q75 = summarize(ratios)

                results.append({
                    "Period": period_name,
                    "Region": region,
                    "LUC": luc_labels[trans],
                    "Unscaled Median (GtCO2)": u_med,
                    "Unscaled Q25 (GtCO2)": u_q25,
                    "Unscaled Q75 (GtCO2)": u_q75,
                    "Scaled Median (GtCO2)": s_med,
                    "Scaled Q25 (GtCO2)": s_q25,
                    "Scaled Q75 (GtCO2)": s_q75,
                    "Overestimation/Underestimation Median (times)": r_med,
                    "Overestimation/Underestimation Q25 (times)": r_q25,
                    "Overestimation/Underestimation Q75 (times)": r_q75,
                })

            all_s_parts = [np.array([v for v in d[t] if np.isfinite(v)], dtype=float) for t in TRANSITION_ORDER]
            all_ur_parts = [np.array([v for v in d[f"unscaled_for_{t}"] if np.isfinite(v)], dtype=float) for t in TRANSITION_ORDER]
            all_s = np.concatenate(all_s_parts) if len(all_s_parts) > 0 else np.array([], dtype=float)
            all_ur = np.concatenate(all_ur_parts) if len(all_ur_parts) > 0 else np.array([], dtype=float)

            a_med, a_q25, a_q75 = summarize(all_s)
            ratios_all = np.array(
                [u / abs(s) for u, s in zip(all_ur, all_s) if np.isfinite(u) and np.isfinite(s) and s != 0],
                dtype=float
            )
            r_med, r_q25, r_q75 = summarize(ratios_all)

            results.append({
                "Period": period_name,
                "Region": region,
                "LUC": luc_labels["all"],
                "Unscaled Median (GtCO2)": u_med,
                "Unscaled Q25 (GtCO2)": u_q25,
                "Unscaled Q75 (GtCO2)": u_q75,
                "Scaled Median (GtCO2)": a_med,
                "Scaled Q25 (GtCO2)": a_q25,
                "Scaled Q75 (GtCO2)": a_q75,
                "Overestimation/Underestimation Median (times)": r_med,
                "Overestimation/Underestimation Q25 (times)": r_q25,
                "Overestimation/Underestimation Q75 (times)": r_q75,
            })

    df = pd.DataFrame(results, columns=col_order)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="TableS8", index=False)

    print(f"[✓] Statistics exported to {output_file.name}")
    return df


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


def plot_periods_subplots(
    upto: Dict[str, Dict[str, List[float]]],
    post: Dict[str, Dict[str, List[float]]]
):
    all_regions = [r for r in R10_REGIONS if upto[r]["unscaled"] or post[r]["unscaled"]]
    if len(all_regions) == 0:
        print("[WARNING] No regional data available for plotting.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(20, 16))

    for ax_idx, (period_label, result_dict) in enumerate([
        ("Up-to Net-Zero", upto),
        ("Post Net-Zero", post),
    ]):
        ax = axes[ax_idx]
        n_regions = len(all_regions)
        x_centers = np.arange(n_regions, dtype=float)

        slot_keys = ["Unscaled"] + TRANSITION_ORDER + ["all"]
        group_width = 0.8
        offset = group_width / len(slot_keys)
        positions = {
            k: x_centers - (group_width / 2) + (offset / 2) + i * offset
            for i, k in enumerate(slot_keys)
        }

        def safe_bp(data_lists, pos_arr, color):
            pairs = [(d, p) for d, p in zip(data_lists, pos_arr) if len(d) > 0 and any(np.isfinite(v) for v in d)]
            if not pairs:
                return
            d_clean, p_clean = zip(*pairs)
            bp = ax.boxplot(
                list(d_clean),
                positions=list(p_clean),
                widths=0.14,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False
            )
            style_bp(bp, color)

        safe_bp(
            [result_dict[r]["unscaled"] for r in all_regions],
            positions["Unscaled"],
            CAT_COLORS["Unscaled"]
        )

        for trans in TRANSITION_ORDER:
            safe_bp(
                [result_dict[r][trans] for r in all_regions],
                positions[trans],
                CAT_COLORS[trans]
            )

        data_all, pos_all = [], []
        for i, r in enumerate(all_regions):
            combined = [v for t in TRANSITION_ORDER for v in result_dict[r][t] if np.isfinite(v)]
            if combined:
                data_all.append(combined)
                pos_all.append(positions["all"][i])

        if data_all:
            bp_all = ax.boxplot(
                data_all,
                positions=pos_all,
                widths=0.14,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False
            )
            style_bp(bp_all, CAT_COLORS["all"])

        ax.set_xticks(x_centers)
        ax.set_xticklabels([r.replace("R10", "") for r in all_regions], rotation=30, ha="center", fontsize=16)
        ax.set_xlim(-1, n_regions)
        ax.set_ylabel("Cumulative Afforestation Carbon (Gt CO$_2$)", fontsize=16, fontweight="bold")
        ax.tick_params(axis="y", labelsize=16)
        ax.set_title(period_label, fontsize=14, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 250)

    legend_elems = [
        mpatches.Patch(facecolor=CAT_COLORS["Unscaled"], edgecolor="none", label="Unscaled"),
        mpatches.Patch(facecolor=CAT_COLORS["nattoaff"], edgecolor="none", label="Natural -> Afforestation"),
        mpatches.Patch(facecolor=CAT_COLORS["agtoaff"], edgecolor="none", label="Agricultural -> Afforestation"),
        mpatches.Patch(facecolor=CAT_COLORS["all"], edgecolor="none", label="All (Combined)"),
    ]
    axes[0].legend(handles=legend_elems, loc="upper right", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(OUTPUT_FIG_PNG, dpi=600, bbox_inches="tight")
    plt.savefig(OUTPUT_FIG_PDF, bbox_inches="tight")
    print(f"[✓] Saved {OUTPUT_FIG_PNG.name} and {OUTPUT_FIG_PDF.name}")
    plt.show()


# --- Main ---
def main():
    print("=" * 80)
    print("REGIONAL AFFORESTATION C1")
    print("=" * 80)

    region_scaling = load_scaling_factors()

    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    aff_ts, co2_ts, year_cols = load_local_data()

    print("=" * 80)
    print("COMPUTING NET-ZERO YEARS")
    print("=" * 80)
    net_zero_dict = build_net_zero_dict(co2_ts)
    nz_years = list(net_zero_dict.values())
    if nz_years:
        print(
            f"Net-zero stats: mean={np.mean(nz_years):.1f}  median={np.median(nz_years):.1f}  "
            f"min={np.min(nz_years):.1f}  max={np.max(nz_years):.1f}  N={len(nz_years)}\n"
        )

    print("=" * 80)
    print("UP-TO NET-ZERO ANALYSIS")
    print("=" * 80)
    upto_dist = gather_cumulative_distributions_per_region(
        aff_ts, year_cols, region_scaling, net_zero_dict, "upto"
    )

    print("\n[*] Up-to net-zero summary:")
    for r in R10_REGIONS:
        d = upto_dist[r]
        note = " <- NO DATA" if not d["unscaled"] else ""
        print(
            f"  {r:20s}: unscaled={len(d['unscaled'])}  "
            f"nattoaff={len(d['nattoaff'])}  agtoaff={len(d['agtoaff'])}{note}"
        )

    print("\n" + "=" * 80)
    print("POST NET-ZERO ANALYSIS")
    print("=" * 80)
    post_dist = gather_cumulative_distributions_per_region(
        aff_ts, year_cols, region_scaling, net_zero_dict, "post"
    )

    print("\n[*] Post net-zero summary:")
    for r in R10_REGIONS:
        d = post_dist[r]
        note = " <- NO DATA" if not d["unscaled"] else ""
        print(
            f"  {r:20s}: unscaled={len(d['unscaled'])}  "
            f"nattoaff={len(d['nattoaff'])}  agtoaff={len(d['agtoaff'])}{note}"
        )

    print("\n" + "=" * 80)
    print("EXPORTING REGIONAL STATISTICS")
    print("=" * 80)
    stats_df = calculate_and_export_statistics(upto_dist, post_dist, OUTPUT_STATS_XLSX)
    print(stats_df.head(20).to_string(index=False))

    print("\n" + "=" * 80)
    print("CREATING VISUALIZATION")
    print("=" * 80)
    plot_periods_subplots(upto_dist, post_dist)

    print("\n" + "=" * 80)
    print("[✓] ALL ANALYSES COMPLETED")
    print("=" * 80)
    print(f"Saved figure: {OUTPUT_FIG_PNG.name}, {OUTPUT_FIG_PDF.name}")
    print(f"Saved stats:  {OUTPUT_STATS_XLSX.name}")


if __name__ == "__main__":
    main()