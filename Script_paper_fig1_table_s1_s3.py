import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyam
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ---------------------------
# Configuration
# ---------------------------
SCENARIO = "SSP1-26"
REGION = "World"

VAR_AG_CROPS = "Agricultural Production|Energy|Crops"  # million tDM/yr
CARBON_FRACTION = 0.485
CO2_TO_C = 44.0 / 12.0

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

CAT_COLORS = {
    "Unscaled": "#66c2a5",
    "AgToBio": "#fc8d62",
    "NatToBio": "#8da0cb",
    "All": "#ffd92f",
}

OUT_COLS = [
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

PER_MODEL_COLS = [
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

# ---------------------------
# Path resolution (GitHub-safe, no local hardcoded path)
# ---------------------------
def resolve_data_dir(required_files):
    candidates = []

    # Optional override:
    # export PAPER_FILES_DIR=/path/to/paper_files
    env_dir = os.getenv("PAPER_FILES_DIR")
    if env_dir:
        p = Path(env_dir).expanduser()
        candidates.append(p)
        candidates.append(p / "paper_files")

    # Notebook/terminal current directory
    cwd = Path.cwd()
    candidates.append(cwd)
    candidates.append(cwd / "paper_files")

    # Script directory
    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
        candidates.append(script_dir / "paper_files")
        candidates.append(script_dir.parent / "paper_files")

    # De-duplicate while preserving order
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
        "Could not locate required input files.\n"
        "Set PAPER_FILES_DIR or place files in a paper_files folder.\n"
        "Expected files:\n"
        + "\n".join(f"  - {f}" for f in required_files)
        + "\nChecked locations:\n"
        + checked
    )


required_files = [
    "scaling_factors_global_bioenergy_carbon_paper.xlsx",
    "AR6_Scenarios_Database_World_v1.1.csv",
    "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx",
]

DATA_DIR = resolve_data_dir(required_files)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SF_FILE = DATA_DIR / "scaling_factors_global_bioenergy_carbon_paper.xlsx"
AR6_DATA_FILE = DATA_DIR / "AR6_Scenarios_Database_World_v1.1.csv"
AR6_META_FILE = DATA_DIR / "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"

# One combined stats file (csv/xlsx), one per-model stats file (csv/xlsx)
OUTPUT_COMBINED_CSV = DATA_DIR / "summary_global_overestimation_biomass_stats_paper_TableS1.csv"
OUTPUT_COMBINED_XLSX = DATA_DIR / "summary_global_overestimation_biomass_stats_paper_TableS1.xlsx"
OUTPUT_PER_MODEL_CSV = DATA_DIR / "summary_global_biomass_per_model_overestimation_stats_paper_TableS3.csv"
OUTPUT_PER_MODEL_XLSX = DATA_DIR / "summary_global_biomass_per_model_overestimation_stats_paper_TableS3.xlsx"

PLOT_PNG = DATA_DIR / "paper_Fig1.png"
PLOT_PDF = DATA_DIR / "paper_Fig1.pdf"

# ---------------------------
# Load scaling factors
# ---------------------------
sf_df = pd.read_excel(SF_FILE)
sf_df.columns = [c.strip() for c in sf_df.columns]
sf_df["IAM_model"] = sf_df["IAM_model"].astype(str).str.strip()
sf_df["Landuse"] = sf_df["Landuse"].astype(str).str.strip().str.lower()
sf_df["scaling_factor"] = pd.to_numeric(sf_df["scaling_factor"], errors="coerce")

# ---------------------------
# Unscaled from Ag crops
# ---------------------------
def compute_unscaled_ag_df():
    data = pyam.IamDataFrame(data=str(AR6_DATA_FILE))
    meta = pd.read_excel(
        AR6_META_FILE,
        sheet_name="meta_Ch3vetted_withclimate"
    ).rename(columns={"Model": "model", "Scenario": "scenario"})
    data.set_meta(meta=meta.set_index(["model", "scenario"]))

    if SCENARIO not in set(data.scenario):
        raise ValueError(f"Scenario '{SCENARIO}' not found.")

    filtered = data.filter(
        scenario=SCENARIO,
        region=REGION,
        variable=VAR_AG_CROPS,
        year=range(YEAR_START, YEAR_END + 1),
    )
    if filtered.data.empty:
        raise ValueError(
            f"No data for variable '{VAR_AG_CROPS}' in scenario '{SCENARIO}' and region '{REGION}'."
        )

    decadal_years = sorted({int(y) for y in filtered.data["year"].unique() if int(y) % 10 == 0})
    filtered = filtered.filter(year=decadal_years)
    ts = filtered.timeseries().reset_index()

    ycols = ts.columns[5:]
    ts[ycols] = ts[ycols].apply(pd.to_numeric, errors="coerce")
    ts[ycols] = ts[ycols] * CARBON_FRACTION * CO2_TO_C

    return ts.set_index(["model", "scenario", "region", "variable", "unit"])


unscaled_df = compute_unscaled_ag_df()


def get_years_from_columns(df, year_max=YEAR_MAX):
    years = []
    for c in df.columns:
        try:
            ci = int(c)
            if ci <= year_max:
                years.append(ci)
        except Exception:
            pass
    return np.array(sorted(set(years)))


def extract_unscaled_gt_per_year(model_name, years):
    sel = unscaled_df.loc[(model_name, SCENARIO, REGION, VAR_AG_CROPS)]
    series = sel if isinstance(sel, pd.Series) else sel.iloc[0]

    keep = []
    for ix in series.index:
        try:
            keep.append(int(ix))
        except Exception:
            continue

    series = series.loc[keep]
    series.index = pd.Index([int(ix) for ix in series.index])
    series = series.sort_index().reindex(years, fill_value=np.nan)

    # MtCO2/yr -> GtCO2/yr
    return years, (series.values / 1000.0)


def cumulative_mt(series: pd.Series, first_year: int, last_year: int) -> float:
    val = pyam.timeseries.cumulative(series, first_year, last_year)
    if val is None or not np.isfinite(val):
        return np.nan
    return float(val)


# ---------------------------
# Build distributions
# ---------------------------
df_models = unscaled_df.index.get_level_values("model").unique()
lower_to_df_model = {m.lower(): m for m in df_models}
years = get_years_from_columns(unscaled_df, YEAR_END)


def build_period_distributions(period_type="upto"):
    model_labels, data_unscaled, data_agtobio, data_nattobio, data_all_luc = [], [], [], [], []

    for model_key in MODELS_ORDER:
        possible_models = [m for m in sf_df["IAM_model"].dropna().unique() if m.lower() == model_key.lower()]
        if not possible_models:
            possible_models = [m for m in sf_df["IAM_model"].dropna().unique() if model_key.lower() in m.lower()]
        if not possible_models:
            print(f"Skipping {model_key}: not in scaling factors.")
            continue

        model_name_in_sf = possible_models[0]
        df_model_name = lower_to_df_model.get(model_key.lower())
        if not df_model_name:
            print(f"Skipping {model_key}: not in unscaled data.")
            continue

        nz_year = int(NET_ZERO_YEARS.get(model_key, YEAR_MAX))
        yrs, y_unscaled = extract_unscaled_gt_per_year(df_model_name, years)
        series = pd.Series(y_unscaled, index=yrs)

        # Non-overlapping split:
        # Up-to: [YEAR_START, nz_year]
        # Post : [nz_year + 1, YEAR_END]
        if period_type == "upto":
            start_y, end_y = YEAR_START, nz_year
        else:
            start_y, end_y = nz_year + 1, YEAR_END

        if start_y > end_y:
            cum_unscaled = 0.0
        else:
            cum_unscaled = cumulative_mt(series, start_y, end_y)

        sf_rows = sf_df[sf_df["IAM_model"].str.lower() == model_name_in_sf.lower()]
        sf_ag = sf_rows[sf_rows["Landuse"] == "agtobio"]["scaling_factor"].values
        sf_nat = sf_rows[sf_rows["Landuse"] == "nattobio"]["scaling_factor"].values

        vals_agtobio = [cum_unscaled * sf for sf in sf_ag if np.isfinite(sf)]
        vals_nattobio = [cum_unscaled * sf for sf in sf_nat if np.isfinite(sf)]

        model_labels.append(model_key)
        data_unscaled.append([cum_unscaled] if np.isfinite(cum_unscaled) else [np.nan])
        data_agtobio.append(vals_agtobio if vals_agtobio else [np.nan])
        data_nattobio.append(vals_nattobio if vals_nattobio else [np.nan])
        data_all_luc.append((vals_agtobio + vals_nattobio) if (vals_agtobio or vals_nattobio) else [np.nan])

    return model_labels, data_unscaled, data_agtobio, data_nattobio, data_all_luc


# ---------------------------
# Pooled summary
# ---------------------------
def summarize_luc(period_label, luc_label, all_unscaled_arr, all_scaled_arr, all_overest_arr):
    u_q25, u_med, u_q75 = (
        np.percentile(all_unscaled_arr, [25, 50, 75]) if len(all_unscaled_arr) > 0 else (np.nan, np.nan, np.nan)
    )
    s_q25, s_med, s_q75 = (
        np.percentile(all_scaled_arr, [25, 50, 75]) if len(all_scaled_arr) > 0 else (np.nan, np.nan, np.nan)
    )
    o_q25, o_med, o_q75 = (
        np.percentile(all_overest_arr, [25, 50, 75]) if len(all_overest_arr) > 0 else (np.nan, np.nan, np.nan)
    )

    def _f(v):
        return float(v) if np.isfinite(v) else np.nan

    return {
        "Period": period_label,
        "LUC": luc_label,
        "Unscaled Median (GtCO2)": _f(u_med),
        "Unscaled Q25 (GtCO2)": _f(u_q25),
        "Unscaled Q75 (GtCO2)": _f(u_q75),
        "Scaled Median (GtCO2)": _f(s_med),
        "Scaled Q25 (GtCO2)": _f(s_q25),
        "Scaled Q75 (GtCO2)": _f(s_q75),
        "Overestimation Median (times)": _f(o_med),
        "Overestimation Q25 (times)": _f(o_q25),
        "Overestimation Q75 (times)": _f(o_q75),
    }


def summarize_period(model_labels, data_unscaled, data_agtobio, data_nattobio, data_all_luc, period_type):
    period_label = "Up-to NZ" if period_type == "upto" else "Post NZ"
    ag_scaled_all, nat_scaled_all, all_scaled_all, unscaled_all = [], [], [], []
    ag_over_all, nat_over_all, all_over_all = [], [], []

    for i in range(len(model_labels)):
        u = np.array(data_unscaled[i], dtype=float); u = u[np.isfinite(u)]
        ag = np.array(data_agtobio[i], dtype=float); ag = ag[np.isfinite(ag)]
        nat = np.array(data_nattobio[i], dtype=float); nat = nat[np.isfinite(nat)]
        all_l = np.array(data_all_luc[i], dtype=float); all_l = all_l[np.isfinite(all_l)]

        if len(u) == 0:
            continue

        unscaled_all.extend(u.tolist())
        ag_scaled_all.extend(ag.tolist())
        nat_scaled_all.extend(nat.tolist())
        all_scaled_all.extend(all_l.tolist())

        ag_over_all.extend([uu / abs(ss) for uu in u for ss in ag if ss != 0])
        nat_over_all.extend([uu / abs(ss) for uu in u for ss in nat if ss != 0])
        all_over_all.extend([uu / abs(ss) for uu in u for ss in all_l if ss != 0])

    rows = [
        summarize_luc(period_label, "AgToBio", np.array(unscaled_all), np.array(ag_scaled_all), np.array(ag_over_all)),
        summarize_luc(period_label, "NatToBio", np.array(unscaled_all), np.array(nat_scaled_all), np.array(nat_over_all)),
        summarize_luc(period_label, "All (Combined)", np.array(unscaled_all), np.array(all_scaled_all), np.array(all_over_all)),
    ]
    return pd.DataFrame(rows, columns=OUT_COLS)


# ---------------------------
# Per-model statistics
# ---------------------------
def _per_model_luc_row(period_label, model_key, luc_label, u_val, scaled_arr):
    def _f(v):
        return float(v) if np.isfinite(v) else np.nan

    scaled_arr = scaled_arr[np.isfinite(scaled_arr)]

    # Single unscaled value per model-period
    u = _f(u_val)

    if len(scaled_arr) > 0:
        s_q25, s_med, s_q75 = np.percentile(scaled_arr, [25, 50, 75])
    else:
        s_q25 = s_med = s_q75 = np.nan

    if len(scaled_arr) > 0 and np.isfinite(u_val) and u_val != 0:
        over = np.array([u_val / abs(s) for s in scaled_arr if s != 0])
        o_q25, o_med, o_q75 = (
            np.percentile(over, [25, 50, 75]) if len(over) > 0 else (np.nan, np.nan, np.nan)
        )
    else:
        o_q25 = o_med = o_q75 = np.nan

    return {
        "Period": period_label,
        "Model": model_key,
        "LUC": luc_label,
        "Unscaled Q25 (GtCO2)": u,
        "Unscaled Median (GtCO2)": u,
        "Unscaled Q75 (GtCO2)": u,
        "Scaled Q25 (GtCO2)": _f(s_q25),
        "Scaled Median (GtCO2)": _f(s_med),
        "Scaled Q75 (GtCO2)": _f(s_q75),
        "Overestimation Q25 (times)": _f(o_q25),
        "Overestimation Median (times)": _f(o_med),
        "Overestimation Q75 (times)": _f(o_q75),
    }


def build_per_model_stats(model_labels, data_unscaled, data_agtobio, data_nattobio, data_all_luc, period_type):
    period_label = "Up-to NZ" if period_type == "upto" else "Post NZ"
    rows = []

    for i, model_key in enumerate(model_labels):
        u_arr = np.array(data_unscaled[i], dtype=float)
        u_arr = u_arr[np.isfinite(u_arr)]
        u_val = u_arr[0] if len(u_arr) > 0 else np.nan

        ag = np.array(data_agtobio[i], dtype=float)
        nat = np.array(data_nattobio[i], dtype=float)
        all_l = np.array(data_all_luc[i], dtype=float)

        rows.append(_per_model_luc_row(period_label, model_key, "AgToBio", u_val, ag))
        rows.append(_per_model_luc_row(period_label, model_key, "NatToBio", u_val, nat))
        rows.append(_per_model_luc_row(period_label, model_key, "All (Combined)", u_val, all_l))

    return pd.DataFrame(rows, columns=PER_MODEL_COLS)


# ---------------------------
# Plotting
# ---------------------------
def _clean_box_data(data_lists):
    cleaned = []
    for vals in data_lists:
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        cleaned.append(arr.tolist() if len(arr) > 0 else [np.nan])
    return cleaned


def _style_bp(bp, color):
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for med in bp["medians"]:
        med.set_color("#333333")
        med.set_linewidth(1.8)
    for w in bp["whiskers"]:
        w.set_color("#555555")
    for c in bp["caps"]:
        c.set_color("#555555")


def plot_period_boxplots(labels_u, data_u, data_ag_u, data_nat_u, data_all_u,
                         labels_p, data_p, data_ag_p, data_nat_p, data_all_p):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("", fontsize=15, fontweight="bold", y=1.03)

    period_cfg = [
        ("Up-to Net-Zero", axes[0], labels_u, data_u, data_ag_u, data_nat_u, data_all_u),
        ("Post Net-Zero", axes[1], labels_p, data_p, data_ag_p, data_nat_p, data_all_p),
    ]

    for title, ax, labels, data_uns, data_ag, data_nat, data_all in period_cfg:
        if len(labels) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue

        x = np.arange(len(labels), dtype=float)
        d = 0.72 / 4.0
        pos_ag = x - 1.5 * d
        pos_nat = x - 0.5 * d
        pos_all = x + 0.5 * d
        pos_uns = x + 1.5 * d

        bp_ag = ax.boxplot(
            _clean_box_data(data_ag), positions=pos_ag, widths=0.16,
            patch_artist=True, showfliers=False, manage_ticks=False
        )
        bp_nat = ax.boxplot(
            _clean_box_data(data_nat), positions=pos_nat, widths=0.16,
            patch_artist=True, showfliers=False, manage_ticks=False
        )
        bp_all = ax.boxplot(
            _clean_box_data(data_all), positions=pos_all, widths=0.16,
            patch_artist=True, showfliers=False, manage_ticks=False
        )

        _style_bp(bp_ag, CAT_COLORS["AgToBio"])
        _style_bp(bp_nat, CAT_COLORS["NatToBio"])
        _style_bp(bp_all, CAT_COLORS["All"])

        for i, vals in enumerate(data_uns):
            arr = np.array(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr) > 0:
                ax.scatter(
                    pos_uns[i], arr[0], marker="D", s=55,
                    color=CAT_COLORS["Unscaled"], zorder=5
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, fontsize=14)
        ax.set_ylabel("Cumulative Bioenergy Carbon (GtCO$_2$)", fontsize=14, fontweight="bold")
        ax.tick_params(axis="y", labelsize=14)
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylim(0, 900)
        ax.set_xlim(-0.8, len(labels) - 0.2)

    legend_elems = [
        mpatches.Patch(facecolor=CAT_COLORS["AgToBio"], edgecolor="none", alpha=0.6, label="AgToBio (scaled)"),
        mpatches.Patch(facecolor=CAT_COLORS["NatToBio"], edgecolor="none", alpha=0.6, label="NatToBio (scaled)"),
        mpatches.Patch(facecolor=CAT_COLORS["All"], edgecolor="none", alpha=0.6, label="All (Combined, scaled)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=CAT_COLORS["Unscaled"],
               markersize=8, linestyle="None", label="Unscaled"),
    ]
    axes[0].legend(handles=legend_elems, loc="upper left", frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=600, bbox_inches="tight")
    plt.savefig(PLOT_PDF, bbox_inches="tight")
    print(f"Saved plots: {PLOT_PNG.name}, {PLOT_PDF.name}")
    plt.show()


# ---------------------------
# Run both periods
# ---------------------------
labels_u, data_u, data_ag_u, data_nat_u, data_all_u = build_period_distributions("upto")
labels_p, data_p, data_ag_p, data_nat_p, data_all_p = build_period_distributions("post")

summary_upto = summarize_period(labels_u, data_u, data_ag_u, data_nat_u, data_all_u, "upto")
summary_post = summarize_period(labels_p, data_p, data_ag_p, data_nat_p, data_all_p, "post")
combined_stats = pd.concat([summary_upto, summary_post], ignore_index=True)

per_model_upto = build_per_model_stats(labels_u, data_u, data_ag_u, data_nat_u, data_all_u, "upto")
per_model_post = build_per_model_stats(labels_p, data_p, data_ag_p, data_nat_p, data_all_p, "post")
per_model_stats = pd.concat([per_model_upto, per_model_post], ignore_index=True)

print("\n" + "=" * 90)
print("COMBINED STATS: UP-TO NET-ZERO")
print("=" * 90)
print(summary_upto.to_string(index=False))

print("\n" + "=" * 90)
print("COMBINED STATS: POST NET-ZERO")
print("=" * 90)
print(summary_post.to_string(index=False))

print("\n" + "=" * 90)
print("PER-MODEL STATS: UP-TO NET-ZERO")
print("=" * 90)
print(per_model_upto.to_string(index=False))

print("\n" + "=" * 90)
print("PER-MODEL STATS: POST NET-ZERO")
print("=" * 90)
print(per_model_post.to_string(index=False))

# ---------------------------
# Save CSVs (one combined + one per-model)
# ---------------------------
combined_stats.to_csv(OUTPUT_COMBINED_CSV, index=False)
per_model_stats.to_csv(OUTPUT_PER_MODEL_CSV, index=False)

# ---------------------------
# Save Excels (one combined + one per-model)
# ---------------------------
with pd.ExcelWriter(OUTPUT_COMBINED_XLSX, engine="openpyxl") as writer:
    summary_upto.to_excel(writer, sheet_name="Up-to NZ", index=False)
    summary_post.to_excel(writer, sheet_name="Post NZ", index=False)
    combined_stats.to_excel(writer, sheet_name="Combined", index=False)

with pd.ExcelWriter(OUTPUT_PER_MODEL_XLSX, engine="openpyxl") as writer:
    per_model_upto.to_excel(writer, sheet_name="Up-to NZ", index=False)
    per_model_post.to_excel(writer, sheet_name="Post NZ", index=False)
    per_model_stats.to_excel(writer, sheet_name="Combined", index=False)

print("\nSaved stats files:")
print(f"  {OUTPUT_COMBINED_CSV.name}")
print(f"  {OUTPUT_COMBINED_XLSX.name}")
print(f"  {OUTPUT_PER_MODEL_CSV.name}")
print(f"  {OUTPUT_PER_MODEL_XLSX.name}")

plot_period_boxplots(
    labels_u, data_u, data_ag_u, data_nat_u, data_all_u,
    labels_p, data_p, data_ag_p, data_nat_p, data_all_p
)