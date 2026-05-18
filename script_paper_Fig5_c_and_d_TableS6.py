#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyam

warnings.filterwarnings("ignore", category=FutureWarning, module=r"pyam\..*")

# --- Configuration ---
CATEGORY = "C1"
REGION = "World"
VAR_AFF_CO2 = "Carbon Sequestration|Land Use|Afforestation"  # MtCO2/yr
VAR_CO2_EMISSIONS = "Emissions|CO2"
YEAR_START = 2020
YEAR_END = 2100
YEAR_MAX = 2100

AR6_WORLD_CSV_NAME = "AR6_Scenarios_Database_World_v1.1.csv"
META_XLSX_NAME = "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
META_SHEET = "meta_Ch3vetted_withclimate"
SF_FILE_NAME = "scaling_factors_afforestation_global_paper.xlsx"

OUTPUT_FIG_BASE_NAME = "Paper_Fig5_c_and_d"
OUTPUT_STATS_XLSX_NAME = "TableS6.xlsx"

DECADAL_YEARS = [y for y in range(YEAR_START, YEAR_END + 1) if y % 10 == 0]
Y_FIXED = None

CAT_COLORS = {
    "Unscaled": "#66c2a5",
    "Natural → Afforestation": "#fc8d62",
    "Agricultural → Afforestation": "#8da0cb",
    "All LUC Combined": "#e78ac3",
}


# --- Path resolution (portable, no local hardcoded path) ---
def resolve_data_dir(required_files: List[str]) -> Path:
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

OUTPUT_STATS_XLSX = DATA_DIR / OUTPUT_STATS_XLSX_NAME
OUTPUT_FIG_PNG = DATA_DIR / f"{OUTPUT_FIG_BASE_NAME}.png"
OUTPUT_FIG_PDF = DATA_DIR / f"{OUTPUT_FIG_BASE_NAME}.pdf"


# --- Helper functions ---
def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


def _pick_col(df: pd.DataFrame, aliases: List[str], required=True, label="column") -> str | None:
    col_map = {_norm_col(c): c for c in df.columns}
    for a in aliases:
        k = _norm_col(a)
        if k in col_map:
            return col_map[k]
    if required:
        raise ValueError(f"Missing {label}. Tried aliases: {aliases}. Available: {list(df.columns)}")
    return None


def normalize_transition(v: str) -> str:
    s = _norm_col(v)
    if s in {"nattoaff", "naturaltoafforestation", "naturaltoaff", "nat2aff"}:
        return "nattoaff"
    if s in {"agtoaff", "agriculturaltoafforestation", "agriculturaltoaff", "ag2aff"}:
        return "agtoaff"
    if s in {"agtonat", "agriculturaltonatural", "agtonatural"}:
        return "agtonat"
    return s


def extract_year_cols(columns, first_year: int, last_year: int, decadal_only: bool = False) -> List[Tuple[int, object]]:
    year_pairs = []
    for c in columns:
        y = None
        if isinstance(c, (int, np.integer)):
            y = int(c)
        elif isinstance(c, str) and c.strip().isdigit():
            y = int(c.strip())

        if y is None:
            continue
        if first_year <= y <= last_year and (not decadal_only or y % 10 == 0):
            year_pairs.append((y, c))

    year_pairs.sort(key=lambda x: x[0])
    return year_pairs


def find_net_zero(years, emissions):
    for i in range(1, len(years)):
        if emissions[i - 1] > 0 and emissions[i] <= 0:
            t0, t1 = years[i - 1], years[i]
            e0, e1 = emissions[i - 1], emissions[i]
            if e0 - e1 != 0:
                frac = e0 / (e0 - e1)
                return t0 + frac * (t1 - t0)
            return t0
    return None


def cumulative_mt(series: pd.Series, first_year: int, last_year: int) -> float:
    if int(first_year) > int(last_year):
        return 0.0
    val = pyam.timeseries.cumulative(series, int(first_year), int(last_year))
    if val is None or not np.isfinite(val):
        return np.nan
    return float(val)


# --- Load scaling factors ---
def load_scaling_factors():
    sf_df = pd.read_excel(SF_FILE)
    sf_df.columns = [str(c).strip() for c in sf_df.columns]

    col_scaling = _pick_col(sf_df, ["scaling_factor", "scalingfactor", "sf"], required=True, label="scaling_factor")
    col_transition = _pick_col(
        sf_df,
        ["Transition", "transition", "landuse", "land_transition", "landtransition", "luc"],
        required=True,
        label="transition",
    )
    col_landmodel = _pick_col(sf_df, ["LandModel", "land_model", "Model", "model", "lsm"], required=False)

    sf_df = sf_df.rename(columns={col_scaling: "scaling_factor", col_transition: "Transition"})
    sf_df["scaling_factor"] = pd.to_numeric(sf_df["scaling_factor"], errors="coerce")

    if col_landmodel:
        sf_df = sf_df.rename(columns={col_landmodel: "LandModel"})
        sf_df["LandModel"] = sf_df["LandModel"].astype(str).str.strip().str.lower()
    else:
        sf_df["LandModel"] = "unknown"
        print("WARNING: Land model column not found; cannot explicitly filter JULES.")

    sf_df["Transition"] = sf_df["Transition"].astype(str).str.strip().map(normalize_transition)
    sf_df = sf_df[np.isfinite(sf_df["scaling_factor"])].copy()


    print(f"Original rows: {len(sf_df)}")
    sf_df = sf_df[sf_df["LandModel"] != "jules"].copy()
    print(f"After filtering JULES: {len(sf_df)}")

    print(f"Before AgToNat filter: {len(sf_df)}")
    sf_df = sf_df[sf_df["Transition"] != "agtonat"].copy()
    print(f"After filtering AgToNat: {len(sf_df)}")
    print(f"Remaining land models: {sorted(sf_df['LandModel'].unique())}")

    sf_nattoaff_all = sf_df[sf_df["Transition"] == "nattoaff"]["scaling_factor"].dropna().to_numpy(dtype=float)
    sf_agtoaff_all = sf_df[sf_df["Transition"] == "agtoaff"]["scaling_factor"].dropna().to_numpy(dtype=float)

    sf_nattoaff_all = sf_nattoaff_all[np.isfinite(sf_nattoaff_all)]
    sf_agtoaff_all = sf_agtoaff_all[np.isfinite(sf_agtoaff_all)]
    sf_combined_all = np.concatenate([sf_nattoaff_all, sf_agtoaff_all])

    print("\nScaling factors (JULES & AgToNat excluded):")
    print(f"  NatToAff: {len(sf_nattoaff_all)}")
    print(f"  AgToAff:  {len(sf_agtoaff_all)}")
    print(f"  Combined (NatToAff + AgToAff): {len(sf_combined_all)}\n")

    return sf_nattoaff_all, sf_agtoaff_all, sf_combined_all


# --- Load local CSV ---
def load_local_data():
    print("Loading AR6 World CSV...")
    iam = pyam.IamDataFrame(data=str(AR6_WORLD_CSV))

    print("Loading metadata...")
    meta = pd.read_excel(META_XLSX, sheet_name=META_SHEET)
    meta = meta.rename(columns={"Model": "model", "Scenario": "scenario"})

    if "Category" not in meta.columns:
        raise ValueError("Column 'Category' not found in metadata sheet.")

    meta["Category"] = meta["Category"].astype(str).str.strip()
    iam.set_meta(meta=meta.set_index(["model", "scenario"]))

    aff = iam.filter(
        Category=CATEGORY,
        variable=VAR_AFF_CO2,
        region=REGION,
        year=DECADAL_YEARS,
    )
    if aff.data.empty:
        raise ValueError(f"No afforestation data found for {CATEGORY}/{REGION}")

    co2 = iam.filter(
        Category=CATEGORY,
        variable=VAR_CO2_EMISSIONS,
        region=REGION,
    )
    if co2.data.empty:
        raise ValueError(f"No CO2 emissions data found for {CATEGORY}/{REGION}")

    aff_meta = aff.meta.reset_index()
    co2_meta = co2.meta.reset_index()

    if not aff_meta["Category"].astype(str).str.strip().eq(CATEGORY).all():
        raise ValueError("Non-C1 scenarios detected in afforestation selection.")
    if not co2_meta["Category"].astype(str).str.strip().eq(CATEGORY).all():
        raise ValueError("Non-C1 scenarios detected in CO2 selection.")

    aff_ts = aff.timeseries()
    co2_ts = co2.timeseries()

    print(f"Afforestation scenarios loaded: {len(aff_ts)} (decadal years only)")
    print(f"CO2 emissions scenarios loaded: {len(co2_ts)} (all years)\n")
    return aff_ts, co2_ts


def build_net_zero_dict(co2_ts):
    net_zero_dict = {}
    idx = co2_ts.index
    models = idx.get_level_values("model")
    scenarios = idx.get_level_values("scenario")

    year_pairs = extract_year_cols(co2_ts.columns, YEAR_START, YEAR_END, decadal_only=False)
    if not year_pairs:
        raise ValueError("No valid CO2 year columns found between YEAR_START and YEAR_END.")

    years = np.array([y for y, _ in year_pairs], dtype=float)
    cols = [c for _, c in year_pairs]

    for i in range(len(co2_ts)):
        model = models[i]
        scenario = scenarios[i]
        emissions = pd.to_numeric(co2_ts.iloc[i][cols], errors="coerce").to_numpy(dtype=float) / 1000.0

        mask = np.isfinite(emissions)
        if np.sum(mask) < 2:
            continue

        nz = find_net_zero(years[mask], emissions[mask])
        if nz is not None and YEAR_START <= nz <= YEAR_END:
            net_zero_dict[(model, scenario)] = float(nz)

    print(f"Net-zero years found for {len(net_zero_dict)} scenarios\n")
    return net_zero_dict


def build_cumulative_distributions(aff_ts, net_zero_dict, sf_nattoaff_all, sf_agtoaff_all, sf_combined_all):
    year_pairs = extract_year_cols(aff_ts.columns, YEAR_START, YEAR_END, decadal_only=True)
    if not year_pairs:
        raise ValueError("No decadal afforestation year columns found in selected range.")

    years_dec = [y for y, _ in year_pairs]
    cols_dec = [c for _, c in year_pairs]

    keys = [
        "unscaled",
        "nattoaff",
        "agtoaff",
        "combined",
        "unscaled_for_nattoaff",
        "unscaled_for_agtoaff",
        "unscaled_for_combined",
    ]
    upto = {k: [] for k in keys}
    post = {k: [] for k in keys}

    split_rows = []

    total = 0
    skipped = 0
    skipped_nz = 0
    idx = aff_ts.index

    for i in range(len(aff_ts)):
        model = idx.get_level_values("model")[i]
        scenario = idx.get_level_values("scenario")[i]

        nz_year = net_zero_dict.get((model, scenario))
        if nz_year is None:
            skipped_nz += 1
            continue

        nz_floor = int(np.floor(nz_year))

        row = aff_ts.iloc[i]
        series = pd.Series(pd.to_numeric(row[cols_dec], errors="coerce").to_numpy(dtype=float), index=years_dec)

        # Non-overlapping split:
        # Up-to = [YEAR_START, NZ_floor]
        # Post  = [NZ_floor + 1, YEAR_END]
        cum_mt_full = cumulative_mt(series, YEAR_START, YEAR_END)
        cum_mt_upto = cumulative_mt(series, YEAR_START, nz_floor)
        cum_mt_post = cumulative_mt(series, nz_floor + 1, YEAR_END)

        split_rows.append(
            {
                "model": model,
                "scenario": scenario,
                "net_zero_year": float(nz_year),
                "net_zero_year_floor": nz_floor,
                "full_2020_2100_GtCO2": (cum_mt_full / 1000.0) if np.isfinite(cum_mt_full) else np.nan,
                "upto_2020_to_nz_GtCO2": (cum_mt_upto / 1000.0) if np.isfinite(cum_mt_upto) else np.nan,
                "post_nzplus1_to_2100_GtCO2": (cum_mt_post / 1000.0) if np.isfinite(cum_mt_post) else np.nan,
                "delta_split_minus_full_GtCO2": (
                    (cum_mt_upto + cum_mt_post - cum_mt_full) / 1000.0
                    if np.isfinite(cum_mt_full) and np.isfinite(cum_mt_upto) and np.isfinite(cum_mt_post)
                    else np.nan
                ),
            }
        )

        cum_gt_upto = cum_mt_upto / 1000.0 if np.isfinite(cum_mt_upto) else np.nan
        cum_gt_post = cum_mt_post / 1000.0 if np.isfinite(cum_mt_post) else np.nan

        if not (np.isfinite(cum_gt_upto) and np.isfinite(cum_gt_post)):
            skipped += 1
            continue

        total += 1

        for period_data, cum_gt in [(upto, cum_gt_upto), (post, cum_gt_post)]:
            period_data["unscaled"].append(cum_gt)
            for sf_arr, key in [
                (sf_nattoaff_all, "nattoaff"),
                (sf_agtoaff_all, "agtoaff"),
                (sf_combined_all, "combined"),
            ]:
                for sf in sf_arr:
                    sv = cum_gt * sf
                    if np.isfinite(sv):
                        period_data[key].append(sv)
                        period_data[f"unscaled_for_{key}"].append(cum_gt)

    split_df = pd.DataFrame(split_rows)

    print(f"Processed {total} scenarios (skipped {skipped_nz} no-NZ, {skipped} invalid)\n")
    if not split_df.empty:
        print("Split consistency delta summary (GtCO2):")
        print(split_df["delta_split_minus_full_GtCO2"].describe())
        print()

    return {"upto": upto, "post": post, "n_scenarios": total, "split_check": split_df}


def calculate_and_export_statistics(all_data):
    rows = []

    transitions = [
        ("nattoaff", "Natural → Afforestation"),
        ("agtoaff", "Agricultural → Afforestation"),
        ("combined", "All LUC Combined"),
    ]

    for period_name, period_data in [("Up-to-Net-Zero", all_data["upto"]), ("Post-Net-Zero", all_data["post"])]:
        unscaled_clean = np.array([v for v in period_data["unscaled"] if np.isfinite(v)], dtype=float)
        if len(unscaled_clean) == 0:
            continue

        for transition_key, transition_label in transitions:
            scaled_vals = np.array([v for v in period_data[transition_key] if np.isfinite(v)], dtype=float)
            unscaled_ref = np.array(period_data[f"unscaled_for_{transition_key}"], dtype=float)

            if len(scaled_vals) == 0:
                continue

            ratios = np.array(
                [u / abs(s) for u, s in zip(unscaled_ref, scaled_vals) if np.isfinite(s) and s != 0],
                dtype=float,
            )

            rows.append(
                {
                    "Period": period_name,
                    "LUC": transition_label,
                    "Unscaled Median (GtCO2)": float(np.median(unscaled_clean)),
                    "Unscaled Q25 (GtCO2)": float(np.percentile(unscaled_clean, 25)),
                    "Unscaled Q75 (GtCO2)": float(np.percentile(unscaled_clean, 75)),
                    "Scaled Median (GtCO2)": float(np.median(scaled_vals)),
                    "Scaled Q25 (GtCO2)": float(np.percentile(scaled_vals, 25)),
                    "Scaled Q75 (GtCO2)": float(np.percentile(scaled_vals, 75)),
                    "Overestimation Median (times)": float(np.median(ratios)) if len(ratios) > 0 else np.nan,
                    "Overestimation Q25 (times)": float(np.percentile(ratios, 25)) if len(ratios) > 0 else np.nan,
                    "Overestimation Q75 (times)": float(np.percentile(ratios, 75)) if len(ratios) > 0 else np.nan,
                }
            )

    col_order = [
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

    stats_df = pd.DataFrame(rows, columns=col_order)

    OUTPUT_STATS_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_STATS_XLSX, engine="openpyxl") as writer:
        stats_df.to_excel(writer, sheet_name="Stats", index=False)
        split_df = all_data.get("split_check")
        if isinstance(split_df, pd.DataFrame) and not split_df.empty:
            split_df.to_excel(writer, sheet_name="SplitConsistency", index=False)

    print(f"Exported statistics to {OUTPUT_STATS_XLSX.name}\n")
    return stats_df


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


def collect_all_values(data_dict):
    values = []
    for period in ["upto", "post"]:
        for key in ["unscaled", "nattoaff", "agtoaff", "combined"]:
            for v in data_dict[period][key]:
                if np.isfinite(v):
                    values.append(float(v))
    return np.array(values, dtype=float)


def plot_combined_multiplot(all_data):
    # Match biomass plotting style
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.linewidth": 1.0,
        }
    )

    if Y_FIXED is not None:
        y_limits = tuple(Y_FIXED)
    else:
        all_vals = collect_all_values(all_data)
        if all_vals.size == 0 or np.all(~np.isfinite(all_vals)):
            y_limits = (0.0, 1.0)
        else:
            vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
            if np.isclose(vmin, vmax):
                pad = max(0.5, abs(vmax) * 0.05)
                y_limits = (vmin - pad, vmax + pad)
            else:
                pad = 0.08 * (vmax - vmin)
                y_limits = (vmin - pad, vmax + pad)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    x_centers = np.array([0, 1.5, 3.0, 4.5])
    width = 0.5

    slot_keys = ["unscaled", "nattoaff", "agtoaff", "combined"]
    slot_colors = [
        CAT_COLORS["Unscaled"],
        CAT_COLORS["Natural → Afforestation"],
        CAT_COLORS["Agricultural → Afforestation"],
        CAT_COLORS["All LUC Combined"],
    ]
    xlabels = [
        "Unscaled",
        "Natural →\nAfforestation",
        "Agricultural →\nAfforestation",
        "All LUC\nCombined",
    ]

    for ax, period_key, title in [
        (ax1, "upto", "Up-to Net-Zero"),
        (ax2, "post", "Post-Net-Zero"),
    ]:
        period_data = all_data[period_key]

        for pos, key, color in zip(x_centers, slot_keys, slot_colors):
            data = period_data[key] if period_data[key] else [np.nan]
            bp = ax.boxplot(
                [data],
                positions=[pos],
                widths=width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
                whis=[5, 95],
                zorder=2,
            )
            style_bp(bp, color)

        ax.set_xlim(-0.5, 5.0)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(xlabels, fontsize=14, fontweight="bold")
        ax.set_ylabel("Cumulative Afforestation Carbon (Gt CO$_2$)", fontsize=16, fontweight="bold")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5, color="#cccccc", zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(y=0, color="black", linewidth=1.0, zorder=2)
        ax.set_ylim(y_limits)
        ax.tick_params(axis="both", labelsize=16, colors="black", width=1.0, length=4)
        ax.tick_params(axis="x", length=0)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(1.0)

    legend_elements = [
        mpatches.Patch(
            facecolor=CAT_COLORS["Unscaled"],
            edgecolor="none",
            alpha=0.8,
            label="Unscaled",
        ),
        mpatches.Patch(
            facecolor=CAT_COLORS["Natural → Afforestation"],
            edgecolor="none",
            alpha=0.8,
            label="Natural → Afforestation",
        ),
        mpatches.Patch(
            facecolor=CAT_COLORS["Agricultural → Afforestation"],
            edgecolor="none",
            alpha=0.8,
            label="Agricultural → Afforestation",
        ),
        mpatches.Patch(
            facecolor=CAT_COLORS["All LUC Combined"],
            edgecolor="none",
            alpha=0.8,
            label="All LUC Combined",
        ),
    ]
    ax2.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=14,
        frameon=True,
        fancybox=False,
        edgecolor="none",
        framealpha=1.0,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUTPUT_FIG_PNG, dpi=600, bbox_inches="tight", facecolor="white")
    plt.savefig(OUTPUT_FIG_PDF, bbox_inches="tight", facecolor="white")
    print(f"Saved {OUTPUT_FIG_PNG.name} and {OUTPUT_FIG_PDF.name}\n")
    plt.show()


# --- Main ---
def main():
    print("=" * 70)
    print("LOADING SCALING FACTORS")
    print("=" * 70)
    sf_nattoaff_all, sf_agtoaff_all, sf_combined_all = load_scaling_factors()

    print("=" * 70)
    print("LOADING DATA FROM AR6 FILES")
    print("=" * 70)
    aff_ts, co2_ts = load_local_data()

    print("=" * 70)
    print("COMPUTING NET-ZERO YEARS")
    print("=" * 70)
    net_zero_dict = build_net_zero_dict(co2_ts)

    print("=" * 70)
    print("BUILDING CUMULATIVE DISTRIBUTIONS")
    print("=" * 70)
    all_data = build_cumulative_distributions(
        aff_ts, net_zero_dict, sf_nattoaff_all, sf_agtoaff_all, sf_combined_all
    )

    print("=" * 70)
    print("CALCULATING AND EXPORTING STATISTICS")
    print("=" * 70)
    calculate_and_export_statistics(all_data)

    print("=" * 70)
    print("CREATING PUBLICATION-QUALITY MULTIPLOT")
    print("=" * 70)
    plot_combined_multiplot(all_data)

    print("\nAll operations completed!")


if __name__ == "__main__":
    main()