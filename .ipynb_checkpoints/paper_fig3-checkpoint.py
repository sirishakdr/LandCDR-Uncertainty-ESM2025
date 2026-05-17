import os
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

CUSTOM_COLORS = ["#003f5c", "#bc5090", "#ffa600"]

# --- Input/output names ---
AR6_WORLD_NAME = "AR6_Scenarios_Database_World_v1.1.csv"
AR6_META_NAME = "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
SCALING_FILE_NAME = "scaling_factors_global_bioenergy_carbon_paper.xlsx"

FIG3_BASE_NAME = "paper_Fig3"
FIG3_STATS_XLSX_NAME = "paper_Fig3_stats.xlsx"


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


DATA_DIR = resolve_data_dir([AR6_WORLD_NAME, AR6_META_NAME, SCALING_FILE_NAME])

AR6_WORLD_FILE = DATA_DIR / AR6_WORLD_NAME
AR6_META_FILE = DATA_DIR / AR6_META_NAME
SF_FILE = DATA_DIR / SCALING_FILE_NAME

FIG3_PNG = DATA_DIR / f"{FIG3_BASE_NAME}.png"
FIG3_PDF = DATA_DIR / f"{FIG3_BASE_NAME}.pdf"
FIG3_STATS_XLSX = DATA_DIR / FIG3_STATS_XLSX_NAME

# --- Load scaling factors ---
sf_df = pd.read_excel(SF_FILE)
sf_df.columns = [c.strip() for c in sf_df.columns]

for col in ["IAM_model", "LandModel", "ESM", "Landuse"]:
    if col in sf_df.columns:
        sf_df[col] = sf_df[col].astype(str).str.strip()

if "scaling_factor" not in sf_df.columns:
    raise ValueError("Column 'scaling_factor' not found in scaling factors file.")
sf_df["scaling_factor"] = pd.to_numeric(sf_df["scaling_factor"], errors="coerce")

# --- Load & compute unscaled (using pyam for filtering/timeseries) ---
def compute_unscaled_df():
    data = pyam.IamDataFrame(data=str(AR6_WORLD_FILE))
    meta = pd.read_excel(
        AR6_META_FILE,
        sheet_name="meta_Ch3vetted_withclimate"
    )
    meta = meta.rename(columns={"Model": "model", "Scenario": "scenario"})
    data.set_meta(meta=meta.set_index(["model", "scenario"]))

    if SCENARIO not in set(data.scenario):
        raise ValueError(f"Scenario '{SCENARIO}' not found.")

    filtered = data.filter(scenario=SCENARIO, region=REGION)
    ag = filtered.filter(variable=VAR_AG_CROPS, year=range(YEAR_START, YEAR_END + 1))
    if ag.data.empty:
        raise ValueError(f"No data found for variable '{VAR_AG_CROPS}'.")

    decadal_years = sorted({int(y) for y in ag.data["year"].unique() if int(y) % 10 == 0})
    ag = ag.filter(year=decadal_years)

    ag_df = ag.timeseries().reset_index()
    ycols = [c for c in ag_df.columns if str(c).isdigit()]
    ag_df[ycols] = ag_df[ycols].apply(pd.to_numeric, errors="coerce")

    # million tDM/yr -> MtCO2/yr
    ag_df[ycols] = ag_df[ycols] * CARBON_FRACTION * CO2_TO_C
    ag_df["unit"] = "MtCO2/yr"

    return ag_df.set_index(["model", "scenario", "region", "variable", "unit"])


Carbon_biomass_modern_total_df = compute_unscaled_df()


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


def extract_unscaled_gt_per_year(df_model: str, years: np.ndarray):
    sel = Carbon_biomass_modern_total_df.loc[(df_model, SCENARIO, REGION, VAR_AG_CROPS)]
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
    # Needed for non-overlapping post window when nz_year == YEAR_MAX
    if int(first_year) > int(last_year):
        return 0.0
    val = pyam.timeseries.cumulative(series, int(first_year), int(last_year))
    if val is None or not np.isfinite(val):
        return np.nan
    return float(val)


# --- Prepare data ---
df_models = Carbon_biomass_modern_total_df.index.get_level_values("model").unique()
lower_to_df_model = {m.lower(): m for m in df_models}
years = get_years_from_columns(Carbon_biomass_modern_total_df, YEAR_END)

# --- Create violin plot data (UP TO NET-ZERO) ---
violin_data = []
unscaled_values = {}
split_consistency = {}

for model_key in MODELS_ORDER:
    possible_models = [m for m in sf_df["IAM_model"].dropna().unique() if m.lower() == model_key.lower()]
    if not possible_models:
        possible_models = [m for m in sf_df["IAM_model"].dropna().unique() if model_key.lower() in m.lower()]
    if not possible_models:
        continue

    model_name_in_sf = possible_models[0]
    df_model_name = lower_to_df_model.get(model_key.lower())
    if not df_model_name:
        continue

    nz_year = int(NET_ZERO_YEARS.get(model_key, YEAR_MAX))
    yrs, y_unscaled = extract_unscaled_gt_per_year(df_model_name, years)
    series = pd.Series(y_unscaled, index=yrs).sort_index()

    # pyam cumulative, up-to window
    cum_unscaled = cumulative_mt(series, YEAR_START, nz_year)
    if not np.isfinite(cum_unscaled):
        continue

    unscaled_values[model_key] = cum_unscaled
    sf_rows = sf_df[sf_df["IAM_model"].str.lower() == model_name_in_sf.lower()]

    for _, row in sf_rows.iterrows():
        scaled_carbon = cum_unscaled * row["scaling_factor"]
        if np.isfinite(scaled_carbon):
            violin_data.append({
                "IAM": model_key,
                "LSM": row["LandModel"],
                "ScaledCarbon": scaled_carbon,
                "UnscaledCarbon": cum_unscaled,
                "ScalingFactor": row["scaling_factor"],
                "Landuse": row["Landuse"],
                "ESM": row["ESM"]
            })

df_violin = pd.DataFrame(violin_data)

# --- Create violin plot data (POST NET-ZERO) ---
violin_data_post_nz = []
unscaled_values_post_nz = {}

for model_key in MODELS_ORDER:
    possible_models = [m for m in sf_df["IAM_model"].dropna().unique() if m.lower() == model_key.lower()]
    if not possible_models:
        possible_models = [m for m in sf_df["IAM_model"].dropna().unique() if model_key.lower() in m.lower()]
    if not possible_models:
        continue

    model_name_in_sf = possible_models[0]
    df_model_name = lower_to_df_model.get(model_key.lower())
    if not df_model_name:
        continue

    nz_year = int(NET_ZERO_YEARS.get(model_key, YEAR_MAX))
    yrs, y_unscaled = extract_unscaled_gt_per_year(df_model_name, years)
    series = pd.Series(y_unscaled, index=yrs).sort_index()

    # pyam cumulative, non-overlapping post window
    cum_unscaled_post_nz = cumulative_mt(series, nz_year + 1, YEAR_MAX)
    if not np.isfinite(cum_unscaled_post_nz):
        continue

    # Split consistency check
    full_total = cumulative_mt(series, YEAR_START, YEAR_MAX)
    upto_total = cumulative_mt(series, YEAR_START, nz_year)
    split_consistency[model_key] = {
        "full_total": full_total,
        "split_sum": upto_total + cum_unscaled_post_nz,
        "delta": (upto_total + cum_unscaled_post_nz) - full_total if np.isfinite(full_total) else np.nan
    }

    unscaled_values_post_nz[model_key] = cum_unscaled_post_nz
    sf_rows = sf_df[sf_df["IAM_model"].str.lower() == model_name_in_sf.lower()]

    for _, row in sf_rows.iterrows():
        scaled_carbon_post_nz = cum_unscaled_post_nz * row["scaling_factor"]
        if np.isfinite(scaled_carbon_post_nz):
            violin_data_post_nz.append({
                "IAM": model_key,
                "LSM": row["LandModel"],
                "ScaledCarbon": scaled_carbon_post_nz,
                "UnscaledCarbon": cum_unscaled_post_nz,
                "ScalingFactor": row["scaling_factor"],
                "Landuse": row["Landuse"],
                "ESM": row["ESM"]
            })

df_violin_post_nz = pd.DataFrame(violin_data_post_nz)

# --- Calculate Statistics ---
def calculate_statistics(df):
    stats_list = []
    for iam in MODELS_ORDER:
        if iam not in df["IAM"].values:
            continue
        data = df[df["IAM"] == iam]["ScaledCarbon"]
        mean_val = data.mean()
        stats_list.append({
            "IAM": iam,
            "Baseline": df[df["IAM"] == iam]["UnscaledCarbon"].iloc[0],
            "Median": data.median(),
            "Mean": mean_val,
            "P5": data.quantile(0.05),
            "P95": data.quantile(0.95),
            "IQR": data.quantile(0.75) - data.quantile(0.25),
            "Range": data.max() - data.min(),
            "Std": data.std(),
            "CV": (data.std() / mean_val) * 100 if mean_val != 0 else np.nan,
            "Count": len(data)
        })
    return pd.DataFrame(stats_list)


stats_df = calculate_statistics(df_violin)
stats_df_post_nz = calculate_statistics(df_violin_post_nz)

# --- Variance Decomposition ---
def variance_decomposition_factorial(df):
    results = []

    for iam in MODELS_ORDER:
        if iam not in df["IAM"].values:
            continue

        iam_data = df[df["IAM"] == iam].copy()
        values = iam_data["ScalingFactor"].values
        grand_mean = values.mean()
        n = len(values)

        SS_total = np.sum((values - grand_mean) ** 2)
        if SS_total == 0:
            continue

        lsm_levels = sorted(iam_data["LSM"].unique())
        esm_levels = sorted(iam_data["ESM"].unique())
        n_lsm = len(lsm_levels)
        n_esm = len(esm_levels)

        lsm_means = {}
        lsm_counts = {}
        for lsm in lsm_levels:
            lsm_data = iam_data[iam_data["LSM"] == lsm]["ScalingFactor"]
            lsm_means[lsm] = lsm_data.mean()
            lsm_counts[lsm] = len(lsm_data)

        esm_means = {}
        esm_counts = {}
        for esm in esm_levels:
            esm_data = iam_data[iam_data["ESM"] == esm]["ScalingFactor"]
            esm_means[esm] = esm_data.mean()
            esm_counts[esm] = len(esm_data)

        cell_means = {}
        cell_counts = {}
        for lsm in lsm_levels:
            for esm in esm_levels:
                cell_data = iam_data[(iam_data["LSM"] == lsm) & (iam_data["ESM"] == esm)]
                if len(cell_data) > 0:
                    cell_means[(lsm, esm)] = cell_data["ScalingFactor"].mean()
                    cell_counts[(lsm, esm)] = len(cell_data)

        SS_LSM = sum(lsm_counts[lsm] * (lsm_means[lsm] - grand_mean) ** 2 for lsm in lsm_levels)
        SS_ESM = sum(esm_counts[esm] * (esm_means[esm] - grand_mean) ** 2 for esm in esm_levels)

        SS_interaction = 0
        for (lsm, esm), cell_mean in cell_means.items():
            n_cell = cell_counts[(lsm, esm)]
            expected_additive = lsm_means[lsm] + esm_means[esm] - grand_mean
            interaction_effect = cell_mean - expected_additive
            SS_interaction += n_cell * (interaction_effect ** 2)

        SS_residual = 0
        for (lsm, esm), cell_mean in cell_means.items():
            cell_data = iam_data[(iam_data["LSM"] == lsm) & (iam_data["ESM"] == esm)]["ScalingFactor"]
            SS_residual += np.sum((cell_data - cell_mean) ** 2)

        var_lsm_pct = (SS_LSM / SS_total) * 100
        var_esm_pct = (SS_ESM / SS_total) * 100
        var_interaction_pct = (SS_interaction / SS_total) * 100
        var_residual_pct = (SS_residual / SS_total) * 100

        df_lsm = n_lsm - 1
        df_esm = n_esm - 1
        df_interaction = (n_lsm - 1) * (n_esm - 1)
        df_residual = n - n_lsm * n_esm

        MS_LSM = SS_LSM / df_lsm if df_lsm > 0 else 0
        MS_ESM = SS_ESM / df_esm if df_esm > 0 else 0
        MS_interaction = SS_interaction / df_interaction if df_interaction > 0 else 0
        MS_residual = SS_residual / df_residual if df_residual > 0 else 0

        F_LSM = MS_LSM / MS_residual if MS_residual > 0 else np.nan
        F_ESM = MS_ESM / MS_residual if MS_residual > 0 else np.nan
        F_interaction = MS_interaction / MS_residual if MS_residual > 0 else np.nan

        p_LSM = 1 - stats.f.cdf(F_LSM, df_lsm, df_residual) if not np.isnan(F_LSM) else np.nan
        p_ESM = 1 - stats.f.cdf(F_ESM, df_esm, df_residual) if not np.isnan(F_ESM) else np.nan
        p_interaction = 1 - stats.f.cdf(F_interaction, df_interaction, df_residual) if not np.isnan(F_interaction) else np.nan

        results.append({
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
        })

    return pd.DataFrame(results)


variance_df = variance_decomposition_factorial(df_violin)
variance_df_post_nz = variance_decomposition_factorial(df_violin_post_nz)


def get_universal_variance(vdf):
    if vdf.empty:
        return {"LSM": np.nan, "ESM": np.nan, "Interaction": np.nan, "Residual": np.nan}
    return {
        "LSM": vdf["Var_LSM"].iloc[0],
        "ESM": vdf["Var_ESM"].iloc[0],
        "Interaction": vdf["Var_Interaction"].iloc[0],
        "Residual": vdf["Var_Residual"].iloc[0]
    }


universal_var = get_universal_variance(variance_df)
universal_var_post_nz = get_universal_variance(variance_df_post_nz)

# --- Save stats as Excel: Fig3 stats ---
split_consistency_df = pd.DataFrame([
    {
        "IAM": iam,
        "full_total": vals["full_total"],
        "split_sum": vals["split_sum"],
        "delta": vals["delta"]
    }
    for iam, vals in split_consistency.items()
])

with pd.ExcelWriter(FIG3_STATS_XLSX, engine="openpyxl") as writer:
    stats_df.to_excel(writer, sheet_name="Up-to NZ Stats", index=False)
    stats_df_post_nz.to_excel(writer, sheet_name="Post NZ Stats", index=False)
    variance_df.to_excel(writer, sheet_name="Up-to NZ Variance", index=False)
    variance_df_post_nz.to_excel(writer, sheet_name="Post NZ Variance", index=False)
    split_consistency_df.to_excel(writer, sheet_name="Split Consistency", index=False)

# ==============================================================================
# FIGURE: Matching Y-Axis Limits
# ==============================================================================

plt.style.use("default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

fig = plt.figure(figsize=(36, 16))
gs = GridSpec(2, 2, figure=fig,
              hspace=0.45, wspace=0.28,
              width_ratios=[3.8, 1.2],
              left=0.05, right=0.97,
              top=0.94,
              bottom=0.10)

lsm_order = ["clm", "jsbach", "jules"]
lsm_colors = {"clm": "#f2c45f", "jsbach": "#bc5090", "jules": "#1a80bb"}
bar_colors_list = ["#e74c3c", "#2c3e50", "#3498db", "#95a5a6"]

max_v1 = df_violin["ScaledCarbon"].max() if not df_violin.empty else np.nan
max_v2 = df_violin_post_nz["ScaledCarbon"].max() if not df_violin_post_nz.empty else np.nan
max_violin = np.nanmax([max_v1, max_v2]) if np.isfinite(np.nanmax([max_v1, max_v2])) else 1.0
max_violin_y = max_violin * 1.15

variances_all = [
    universal_var["LSM"], universal_var["ESM"], universal_var["Interaction"], universal_var["Residual"],
    universal_var_post_nz["LSM"], universal_var_post_nz["ESM"], universal_var_post_nz["Interaction"], universal_var_post_nz["Residual"]
]
max_variance = np.nanmax(variances_all) if np.isfinite(np.nanmax(variances_all)) else 1.0
max_variance_y = max_variance * 1.42


def plot_violin_and_bar(fig, gs_row, col_offset, df_violin_data, unscaled_vals, variance_df_data,
                        universal_var_data, title_prefix, variance_title, panel_start_letter, show_legend=False,
                        violin_ylim=None, variance_ylim=None):
    # Violin plot
    ax_violin = fig.add_subplot(gs[gs_row, col_offset])
    ax_violin.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=0.8, zorder=0)
    ax_violin.set_axisbelow(True)
    ax_violin.set_facecolor("#fafafa")

    for iam_idx, iam in enumerate(MODELS_ORDER):
        if iam not in df_violin_data["IAM"].values:
            continue

        iam_data = df_violin_data[df_violin_data["IAM"] == iam]

        violin_width = 0.18
        total_width = 0.9
        spacing = total_width / 4
        start_offset = -total_width / 2 + spacing / 2

        for lsm_idx, lsm in enumerate(lsm_order):
            lsm_data = iam_data[iam_data["LSM"] == lsm]["ScaledCarbon"].values
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
                bw_method="scott"
            )

            for pc in parts["bodies"]:
                pc.set_facecolor(lsm_colors[lsm])
                pc.set_alpha(0.65)
                pc.set_edgecolor("white")
                pc.set_linewidth(1)

            q1, median, q3 = np.percentile(lsm_data, [25, 50, 75])
            ax_violin.plot([position, position], [q1, q3], color="#1a1a1a", linewidth=3, alpha=0.85, zorder=10)
            ax_violin.plot([position - 0.08, position + 0.08], [median, median], color="#1a1a1a", linewidth=3, alpha=0.85, zorder=10)

        all_data = iam_data["ScaledCarbon"].values
        overall_position = iam_idx + start_offset + 3 * spacing

        parts_overall = ax_violin.violinplot(
            [all_data],
            positions=[overall_position],
            widths=violin_width,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method="scott"
        )

        for pc in parts_overall["bodies"]:
            pc.set_facecolor("#a00000")
            pc.set_alpha(0.65)
            pc.set_edgecolor("white")
            pc.set_linewidth(1)

        q1_overall, median_overall, q3_overall = np.percentile(all_data, [25, 50, 75])
        ax_violin.plot([overall_position, overall_position], [q1_overall, q3_overall],
                       color="#8B0000", linewidth=3, alpha=0.9, zorder=10)
        ax_violin.plot([overall_position - 0.08, overall_position + 0.08], [median_overall, median_overall],
                       color="#8B0000", linewidth=3, alpha=0.9, zorder=10)

        if iam in unscaled_vals:
            unscaled_value = unscaled_vals[iam]
            diamond_position = iam_idx
            ax_violin.plot(diamond_position, unscaled_value, marker="D", markersize=16,
                           color="#00C853", markerfacecolor="#00E676", markeredgecolor="#00691B",
                           markeredgewidth=3, zorder=15)

    for iam_idx in range(len(MODELS_ORDER) - 1):
        separator_x = iam_idx + 0.5
        ax_violin.axvline(x=separator_x, color="#cccccc", linestyle="-", linewidth=2.5, alpha=0.4, zorder=1)

    ax_violin.set_xticks(range(len(MODELS_ORDER)))
    ax_violin.set_xticklabels(MODELS_ORDER, fontsize=22, rotation=25, ha="center", fontweight="bold")
    ax_violin.set_xlabel("", fontsize=22, fontweight="bold", labelpad=12)
    ax_violin.set_ylabel("Cumulative Bioenergy Carbon (GtCO$_2$)", fontsize=22, fontweight="bold", labelpad=12)
    ax_violin.set_title(f"({chr(97 + panel_start_letter)}) {title_prefix}", fontsize=26, fontweight="bold", pad=18, loc="left")

    if violin_ylim:
        ax_violin.set_ylim(violin_ylim)

    if show_legend:
        legend_elements = [
            mpatches.Patch(facecolor="#f2c45f", label="CLM", alpha=0.7, edgecolor="white", linewidth=1),
            mpatches.Patch(facecolor="#bc5090", label="JSBACH", alpha=0.7, edgecolor="white", linewidth=1),
            mpatches.Patch(facecolor="#1a80bb", label="JULES", alpha=0.7, edgecolor="white", linewidth=1),
            mpatches.Patch(facecolor="#a00000", label="Overall", alpha=0.7, edgecolor="white", linewidth=1),
            mpatches.Patch(facecolor="#00E676", label="Unscaled", alpha=0.85, edgecolor="#00691B", linewidth=2),
        ]
        ax_violin.legend(handles=legend_elements, loc="upper left", frameon=True, framealpha=0.98,
                         fontsize=22, title_fontsize=20, edgecolor="#cccccc", ncol=3,
                         handlelength=2.2, handleheight=0.8, fancybox=True, shadow=True)

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

    sources = ["LSM", "ESM", "LSM×ESM\nInteraction", "LUC\n(Residual)"]
    variances = [universal_var_data["LSM"], universal_var_data["ESM"],
                 universal_var_data["Interaction"], universal_var_data["Residual"]]

    if variance_df_data.empty:
        p_values = [np.nan, np.nan, np.nan, np.nan]
    else:
        p_values = [
            variance_df_data["p_LSM"].iloc[0],
            variance_df_data["p_ESM"].iloc[0],
            variance_df_data["p_Interaction"].iloc[0],
            np.nan
        ]

    x_positions = np.arange(len(sources))
    for i, (x_pos, var, color) in enumerate(zip(x_positions, variances, bar_colors_list)):
        ax_var.bar(x_pos, var, color=color, alpha=0.88, edgecolor="#1a1a1a", linewidth=1.8, width=0.45)

        if np.isfinite(var):
            ax_var.text(x_pos, var + 3, f"{var:.1f}%", ha="center", va="bottom", fontsize=20, fontweight="bold", color="#1a1a1a")

        if not np.isnan(p_values[i]) and np.isfinite(var):
            if p_values[i] < 0.001:
                sig_text = "***"
            elif p_values[i] < 0.01:
                sig_text = "**"
            elif p_values[i] < 0.05:
                sig_text = "*"
            else:
                sig_text = "n.s."
            ax_var.text(x_pos, var + 9, sig_text, ha="center", va="bottom", fontsize=20,
                        style="italic", fontweight="bold", color="#d32f2f")

    ax_var.set_xticks(x_positions)
    ax_var.set_xticklabels(sources, fontsize=22, fontweight="bold")
    ax_var.set_ylabel("Variance (%)", fontsize=22, fontweight="bold", labelpad=10)
    ax_var.set_title(f"({chr(98 + panel_start_letter)}) {variance_title}",
                     fontsize=22, fontweight="bold", pad=15, loc="left")

    if variance_ylim:
        ax_var.set_ylim([0, variance_ylim])

    if show_legend:
        legend_patches = [
            mpatches.Patch(color=bar_colors_list[0], label="LSM", alpha=0.88, edgecolor="#1a1a1a", linewidth=1.5),
            mpatches.Patch(color=bar_colors_list[1], label="ESM", alpha=0.88, edgecolor="#1a1a1a", linewidth=1.5),
            mpatches.Patch(color=bar_colors_list[2], label="Interaction", alpha=0.88, edgecolor="#1a1a1a", linewidth=1.5),
            mpatches.Patch(color=bar_colors_list[3], label="LUC", alpha=0.88, edgecolor="#1a1a1a", linewidth=1.5),
        ]
        ax_var.legend(handles=legend_patches, loc="upper right", frameon=True, framealpha=0.98,
                      fontsize=22, edgecolor="#cccccc", fancybox=True, shadow=True)

    ax_var.text(0.02, 0.97, "*** p<0.001 | ** p<0.01 | * p<0.05 | n.s.",
                transform=ax_var.transAxes, ha="left", va="top", fontsize=22, style="italic",
                bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffeb", alpha=0.92,
                          edgecolor="#cccccc", linewidth=1.2), color="#1a1a1a", fontweight="bold")

    ax_var.spines["top"].set_visible(False)
    ax_var.spines["right"].set_visible(False)
    ax_var.spines["left"].set_linewidth(2.5)
    ax_var.spines["bottom"].set_linewidth(2.5)
    ax_var.tick_params(axis="both", labelsize=22, width=2, length=5)


# Plot up-to net-zero (row 0) - NO LEGENDS
plot_violin_and_bar(
    fig, 0, 0, df_violin, unscaled_values, variance_df, universal_var,
    "Up-to Net-Zero", "Uncertainty Quantification (Factorial ANOVA: Up-to Net-Zero)", 0, show_legend=False,
    violin_ylim=[0, max_violin_y], variance_ylim=max_variance_y
)

# Plot post net-zero (row 1) - WITH LEGENDS
plot_violin_and_bar(
    fig, 1, 0, df_violin_post_nz, unscaled_values_post_nz, variance_df_post_nz,
    universal_var_post_nz, "Post-Net-Zero", "Uncertainty Quantification (Factorial ANOVA: Post-Net-Zero)", 2, show_legend=True,
    violin_ylim=[0, max_violin_y], variance_ylim=max_variance_y
)

fig.suptitle("", fontsize=26, fontweight="bold", y=0.96, x=0.5, ha="center")

plt.savefig(FIG3_PNG, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.savefig(FIG3_PDF, bbox_inches="tight", facecolor="white", edgecolor="none")

print("\n" + "=" * 80)
print("FIGURE CREATED SUCCESSFULLY")
print("=" * 80)

print("\nNet-zero split consistency check (delta should be near 0):")
for iam, vals in split_consistency.items():
    print(f"  {iam}: full={vals['full_total']:.4f}, split_sum={vals['split_sum']:.4f}, delta={vals['delta']:.6e}")

print("\nFiles saved:")
print(f"   - {FIG3_PNG.name}")
print(f"   - {FIG3_PDF.name}")
print(f"   - {FIG3_STATS_XLSX.name}")

plt.show()