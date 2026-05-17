import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pyam
from pathlib import Path

# --- Configuration ---
CATEGORY = 'C1'
REGION = 'World'
VAR_AG_CROPS = "Agricultural Production|Energy|Crops"  # million tDM/yr
CARBON_FRACTION = 0.485  # tC per tDM
CO2_TO_C = 44.0 / 12.0  # tCO2 per tC
VAR_CO2_EMISSIONS = "Emissions|CO2"
YEAR_START = 2020
YEAR_END = 2100
YEAR_MAX = 2100

META_SHEET = "meta_Ch3vetted_withclimate"
DECADAL_YEARS = [y for y in range(YEAR_START, YEAR_END + 1) if y % 10 == 0]

CAT_COLORS = {
    "Unscaled": "#66c2a5",
    "Agric → Bioenergy": "#fc8d62",
    "Natural → Bioenergy": "#8da0cb",
    "All LUC Combined": "#e78ac3",
}

# --- File names ---
AR6_WORLD_CSV_NAME = "AR6_Scenarios_Database_World_v1.1.csv"
META_XLSX_NAME = "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
SF_FILE_NAME = "scaling_factors_global_bioenergy_carbon_paper.xlsx"

STATS_CSV_NAME = "paper_tableS5.csv"
PLOT_PNG_NAME = "Fig5_a_b_paper.png"
PLOT_PDF_NAME = "Fig5_a_b_paper.pdf"


# --- DATA_DIR resolver (portable, GitHub-safe, no local hardcoded path) ---
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

    # Script directory (when run as .py)
    if "__file__" in globals():
        script_dir = Path(__file__).resolve().parent
        candidates.append(script_dir)
        candidates.append(script_dir / "paper_files")
        candidates.append(script_dir.parent / "paper_files")

    # De-duplicate in order
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


required_inputs = [AR6_WORLD_CSV_NAME, META_XLSX_NAME, SF_FILE_NAME]
DATA_DIR = resolve_data_dir(required_inputs)
DATA_DIR.mkdir(parents=True, exist_ok=True)

AR6_WORLD_CSV = DATA_DIR / AR6_WORLD_CSV_NAME
META_XLSX = DATA_DIR / META_XLSX_NAME
SF_FILE = DATA_DIR / SF_FILE_NAME

STATS_CSV = DATA_DIR / STATS_CSV_NAME
PLOT_PNG = DATA_DIR / PLOT_PNG_NAME
PLOT_PDF = DATA_DIR / PLOT_PDF_NAME


# --- Load scaling factors ---
sf_df = pd.read_excel(SF_FILE)
sf_df.columns = [c.strip() for c in sf_df.columns]
sf_df['IAM_model'] = sf_df['IAM_model'].astype(str).str.strip().str.lower()
if 'LandModel' in sf_df.columns:
    sf_df['LandModel'] = sf_df['LandModel'].astype(str).str.strip().str.lower()
if 'ESM' in sf_df.columns:
    sf_df['ESM'] = sf_df['ESM'].astype(str).str.strip().str.lower()
if 'Landuse' in sf_df.columns:
    sf_df['Landuse'] = sf_df['Landuse'].astype(str).str.strip().str.lower()
if 'scaling_factor' in sf_df.columns:
    sf_df['scaling_factor'] = pd.to_numeric(sf_df['scaling_factor'], errors='coerce')

print(f"✓ Loaded scaling factors: {len(sf_df)} rows")

sf_agtobio_all = sf_df[sf_df['Landuse'] == 'agtobio']['scaling_factor'].dropna().values
sf_nattobio_all = sf_df[sf_df['Landuse'] == 'nattobio']['scaling_factor'].dropna().values
sf_agtobio_all = sf_agtobio_all[np.isfinite(sf_agtobio_all)]
sf_nattobio_all = sf_nattobio_all[np.isfinite(sf_nattobio_all)]

print("\nScaling factors:")
print(f"  AgtoBio:  {len(sf_agtobio_all)}")
print(f"  NatToBio: {len(sf_nattobio_all)}\n")


# --- Load AR6 + metadata ---
def load_local_data():
    print("Loading AR6 World CSV...")
    iam = pyam.IamDataFrame(data=str(AR6_WORLD_CSV))

    print("Loading metadata...")
    meta = pd.read_excel(META_XLSX, sheet_name=META_SHEET)
    meta = meta.rename(columns={"Model": "model", "Scenario": "scenario"})
    meta['Category'] = meta['Category'].astype(str).str.strip()
    iam.set_meta(meta=meta.set_index(["model", "scenario"]))

    # Explicit C1 info (for reporting)
    c1_pairs = meta[meta['Category'] == CATEGORY][['model', 'scenario']].drop_duplicates()
    print(f"C1 scenarios in metadata: {len(c1_pairs)}")

    # Agricultural crops — decadal years only
    ag_crops = iam.filter(
        Category=CATEGORY,
        variable=VAR_AG_CROPS,
        region=REGION,
        year=DECADAL_YEARS
    )
    if ag_crops.data.empty:
        raise ValueError(f"No data found for {VAR_AG_CROPS}")

    # CO2 emissions — all years for accurate net-zero interpolation
    co2 = iam.filter(
        Category=CATEGORY,
        variable=VAR_CO2_EMISSIONS,
        region=REGION
    )
    if co2.data.empty:
        raise ValueError("No CO2 emissions data found")

    # Strict runtime validation: only C1
    ag_meta = ag_crops.meta.reset_index()
    co2_meta = co2.meta.reset_index()
    if not ag_meta["Category"].astype(str).str.strip().eq(CATEGORY).all():
        raise ValueError("Non-C1 scenarios detected in agricultural crops selection.")
    if not co2_meta["Category"].astype(str).str.strip().eq(CATEGORY).all():
        raise ValueError("Non-C1 scenarios detected in CO2 selection.")

    ag_ts = ag_crops.timeseries()
    co2_ts = co2.timeseries()

    # Convert million tDM/yr -> million tCO2/yr
    ag_ts = ag_ts * CARBON_FRACTION * CO2_TO_C

    print(f"Agricultural crops rows: {len(ag_ts)}")
    print(f"CO2 rows:                {len(co2_ts)} (all years)\n")
    return ag_ts, co2_ts


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
    val = pyam.timeseries.cumulative(series, first_year, last_year)
    if val is None or not np.isfinite(val):
        return np.nan
    return float(val)


def build_net_zero_dict(co2_ts):
    # Full year resolution — no decadal filter
    net_zero_dict = {}
    idx = co2_ts.index
    year_cols = sorted(
        [c for c in co2_ts.columns if isinstance(c, (int, np.integer)) and YEAR_START <= c <= YEAR_END]
    )
    for i in range(len(co2_ts)):
        model = idx.get_level_values('model')[i]
        scenario = idx.get_level_values('scenario')[i]
        emissions = co2_ts.iloc[i][year_cols].values.astype(float) / 1000.0
        nz = find_net_zero(np.array(year_cols), emissions)
        if nz and YEAR_START <= nz <= YEAR_END:
            net_zero_dict[(model, scenario)] = nz
    print(f"Net-zero years found for {len(net_zero_dict)} scenarios\n")
    return net_zero_dict


def build_cumulative_distributions(ag_ts, year_cols, net_zero_dict):
    # Decadal year columns only for integration
    keys = [
        'unscaled', 'agtobio', 'nattobio', 'all_luc',
        'unscaled_for_agtobio', 'unscaled_for_nattobio', 'unscaled_for_all_luc'
    ]
    upto = {k: [] for k in keys}
    post = {k: [] for k in keys}

    total = skipped = skipped_nz = 0
    idx = ag_ts.index

    for i in range(len(ag_ts)):
        model = idx.get_level_values('model')[i]
        scenario = idx.get_level_values('scenario')[i]

        nz_year = net_zero_dict.get((model, scenario))
        if nz_year is None:
            skipped_nz += 1
            continue

        nz_int = int(nz_year)

        row = ag_ts.iloc[i]
        series = pd.Series(row[year_cols].values.astype(float), index=year_cols)

        # Non-overlapping split:
        # Up-to: [YEAR_START, nz_int]
        # Post:  [nz_int + 1, YEAR_END]
        cum_mt_upto = cumulative_mt(series, YEAR_START, nz_int)
        if nz_int + 1 <= YEAR_END:
            cum_mt_post = cumulative_mt(series, nz_int + 1, YEAR_END)
        else:
            cum_mt_post = np.nan

        cum_gt_upto = cum_mt_upto / 1000.0 if np.isfinite(cum_mt_upto) else np.nan
        cum_gt_post = cum_mt_post / 1000.0 if np.isfinite(cum_mt_post) else np.nan

        valid_upto = np.isfinite(cum_gt_upto) and cum_gt_upto > 0
        valid_post = np.isfinite(cum_gt_post) and cum_gt_post > 0

        if not (valid_upto or valid_post):
            skipped += 1
            continue

        total += 1

        for period_data, cum_gt, valid in [
            (upto, cum_gt_upto, valid_upto),
            (post, cum_gt_post, valid_post),
        ]:
            if not valid:
                continue
            period_data['unscaled'].append(cum_gt)
            for sf_arr, key in [
                (sf_agtobio_all, 'agtobio'),
                (sf_nattobio_all, 'nattobio'),
            ]:
                for sf in sf_arr:
                    sv = cum_gt * sf
                    if np.isfinite(sv) and sv > 0:
                        period_data[key].append(sv)
                        period_data['all_luc'].append(sv)
                        period_data[f'unscaled_for_{key}'].append(cum_gt)
                        period_data['unscaled_for_all_luc'].append(cum_gt)

    print(f"Processed {total} scenarios (skipped {skipped_nz} no-NZ, {skipped} invalid)")
    print(
        f"\nUp-to Net-Zero:  unscaled={len(upto['unscaled'])}  "
        f"agtobio={len(upto['agtobio'])}  nattobio={len(upto['nattobio'])}  all_luc={len(upto['all_luc'])}"
    )
    print(
        f"Post Net-Zero:   unscaled={len(post['unscaled'])}  "
        f"agtobio={len(post['agtobio'])}  nattobio={len(post['nattobio'])}  all_luc={len(post['all_luc'])}\n"
    )
    return {'upto': upto, 'post': post, 'n_scenarios': total}


def calculate_and_export_statistics_csv(all_data):
    csv_rows = []

    transitions = [
        ('agtobio', 'Agricultural to Bioenergy'),
        ('nattobio', 'Natural land to Bioenergy'),
        ('all_luc', 'All LUC Combined'),
    ]

    for period_name, period_data in [
        ('Up-to-Net-Zero', all_data['upto']),
        ('Post-Net-Zero', all_data['post']),
    ]:
        unscaled_clean = np.array([v for v in period_data['unscaled'] if np.isfinite(v)])

        for transition, label in transitions:
            scaled_vals = np.array([v for v in period_data[transition] if np.isfinite(v)])
            unscaled_ref = np.array(period_data[f'unscaled_for_{transition}'])

            if len(scaled_vals) == 0:
                continue

            ratios = np.array([
                u / abs(s) for u, s in zip(unscaled_ref, scaled_vals)
                if np.isfinite(s) and s != 0
            ])

            csv_rows.append({
                'Period': period_name,
                'LUC': label,
                'Unscaled Median (GtCO2)': np.median(unscaled_clean),
                'Unscaled Q25 (GtCO2)': np.percentile(unscaled_clean, 25),
                'Unscaled Q75 (GtCO2)': np.percentile(unscaled_clean, 75),
                'Scaled Median (GtCO2)': np.median(scaled_vals),
                'Scaled Q25 (GtCO2)': np.percentile(scaled_vals, 25),
                'Scaled Q75 (GtCO2)': np.percentile(scaled_vals, 75),
                'Overestimation Median (times)': np.median(ratios) if len(ratios) > 0 else np.nan,
                'Overestimation Q25 (times)': np.percentile(ratios, 25) if len(ratios) > 0 else np.nan,
                'Overestimation Q75 (times)': np.percentile(ratios, 75) if len(ratios) > 0 else np.nan,
            })

    csv_df = pd.DataFrame(csv_rows)
    csv_df.to_csv(STATS_CSV, index=False)
    print(f"✓ Exported statistics to {STATS_CSV.name}\n")

    print("=" * 140)
    print("STATISTICAL SUMMARY - Biomass Carbon from Agricultural Energy Crops (Gt CO2, C1 only)")
    print("=" * 140)
    for _, row in csv_df.iterrows():
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
            f"Q25={row['Overestimation Q25 (times)']:>7.2f}x  "
            f"Q75={row['Overestimation Q75 (times)']:>7.2f}x"
        )
    print("=" * 140 + "\n")
    return csv_df


def style_bp(bp, color):
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor('none')
        patch.set_linewidth(1.2)
    for med in bp['medians']:
        med.set_color('black')
        med.set_linewidth(2.0)
        med.set_linestyle('-')
    for w in bp['whiskers']:
        w.set_color('black')
        w.set_linewidth(1.0)
        w.set_linestyle('-')
    for c in bp['caps']:
        c.set_color('black')
        c.set_linewidth(1.0)


def plot_combined_multiplot(all_data):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10, 'axes.linewidth': 1.0,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    x_centers = np.array([0, 1.5, 3.0, 4.5])
    width = 0.5

    slot_keys = ['unscaled', 'agtobio', 'nattobio', 'all_luc']
    slot_colors = [
        CAT_COLORS['Unscaled'],
        CAT_COLORS['Agric → Bioenergy'],
        CAT_COLORS['Natural → Bioenergy'],
        CAT_COLORS['All LUC Combined']
    ]
    xlabels = [
        'Unscaled',
        'Agricultural →\nBioenergy',
        'Natural →\nBioenergy',
        'All LUC\nCombined'
    ]

    for ax, period_key, title in [
        (ax1, 'upto', 'Up-to Net-Zero'),
        (ax2, 'post', 'Post-Net-Zero'),
    ]:
        period_data = all_data[period_key]
        for pos, key, color in zip(x_centers, slot_keys, slot_colors):
            data = period_data[key] if period_data[key] else [np.nan]
            bp = ax.boxplot(
                [data], positions=[pos], widths=width,
                patch_artist=True, manage_ticks=False,
                showfliers=False, whis=[5, 95], zorder=2
            )
            style_bp(bp, color)

        ax.set_xlim(-0.5, 5.0)
        ax.set_xticks(x_centers)
        ax.set_xticklabels(xlabels, fontsize=14, fontweight='bold')
        ax.set_ylabel("Cumulative Bioenergy Carbon (GtCO$_2$)", fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc', zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(y=0, color='black', linewidth=1.0, zorder=2)
        ax.set_ylim(0, 1500)
        ax.tick_params(axis='both', labelsize=16, colors='black', width=1.0, length=4)
        ax.tick_params(axis='x', length=0)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_color('black')
            ax.spines[spine].set_linewidth(1.0)

    legend_elements = [
        mpatches.Patch(facecolor=CAT_COLORS['Unscaled'], edgecolor='none', alpha=0.8, label='Unscaled'),
        mpatches.Patch(facecolor=CAT_COLORS['Agric → Bioenergy'], edgecolor='none', alpha=0.8, label='Agricultural → Bioenergy'),
        mpatches.Patch(facecolor=CAT_COLORS['Natural → Bioenergy'], edgecolor='none', alpha=0.8, label='Natural → Bioenergy'),
        mpatches.Patch(facecolor=CAT_COLORS['All LUC Combined'], edgecolor='none', alpha=0.8, label='All LUC Combined'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=14,
               frameon=True, fancybox=False, edgecolor='none', framealpha=1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(PLOT_PNG, dpi=600, bbox_inches='tight', facecolor='white')
    plt.savefig(PLOT_PDF, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved plot files: {PLOT_PNG.name}, {PLOT_PDF.name}\n")
    plt.show()


# --- Main ---
print("=" * 70)
print("LOADING DATA (C1 only)")
print("=" * 70)
ag_ts, co2_ts = load_local_data()

print("=" * 70)
print("COMPUTING NET-ZERO YEARS (full year resolution)")
print("=" * 70)
net_zero_dict = build_net_zero_dict(co2_ts)

nz_years = list(net_zero_dict.values())
print(
    f"Net-zero year stats: Mean={np.mean(nz_years):.1f}  Median={np.median(nz_years):.1f}  "
    f"Min={np.min(nz_years):.1f}  Max={np.max(nz_years):.1f}\n"
)

print("=" * 70)
print("BUILDING CUMULATIVE DISTRIBUTIONS (C1, decadal years)")
print("=" * 70)
year_cols = sorted([
    c for c in ag_ts.columns
    if isinstance(c, (int, np.integer)) and YEAR_START <= c <= YEAR_END and c % 10 == 0
])
all_data = build_cumulative_distributions(ag_ts, year_cols, net_zero_dict)

print("=" * 70)
print("CALCULATING AND EXPORTING STATISTICS")
print("=" * 70)
calculate_and_export_statistics_csv(all_data)

print("=" * 70)
print("CREATING PUBLICATION-QUALITY MULTIPLOT")
print("=" * 70)
plot_combined_multiplot(all_data)

print(f"\n✓ All operations completed! ({all_data['n_scenarios']} C1 scenarios processed)")