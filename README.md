This repository is linked to the following research paper prepared for peer-review: Sirisha Kalidindi, Gabriel Abrahão, Anna B. Harper, Michael Crawford, Jens Heinke, Elmar Kriegler, Joeri Rogelj (2026). Quantifying uncertainty in land-based carbon removals in AR6 mitigation pathways.

The repository includes Python scripts to reproduce figures and analyses from the paper.

## Install from GitHub

To clone this repository, run:

```bash
git clone git@github.com:sirishakdr/LandCDR-Uncertainty-ESM2025.git
cd LandCDR-Uncertainty-ESM2025
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate your_env_name
```
## Data Requirements

Datasets used in the study and required by the Python scripts are available from:

- [Zenodo](https://zenodo.org/uploads/20309536)
- [AR6 Scenarios Database metadata indicators v1.1](https://data.ece.iiasa.ac.at/ar6/#/downloads/AR6_Scenarios_Database_metadata_indicators_v1.1)
- [AR6 Scenarios Database World v1.1](https://data.ece.iiasa.ac.at/ar6/#/downloads/AR6_Scenarios_Database_World_v1.1)
- [AR6 Scenarios Database R10 regions v1.1](https://data.ece.iiasa.ac.at/ar6/#/downloads/AR6_Scenarios_Database_R10_regions_v1.1)
- [R10 regions mask](https://doi.org/10.5281/zenodo.8362562)

Place required input files in a `paper_files` folder inside the cloned repository root before running scripts.


## Run Python Scripts

Run the scripts from the repository root after activating the Conda environment and placing required datasets in the `paper_files` folder.

```bash
python Script_paper_fig1_table_s1_s3.py
python script_paper_Fig2_TableS2_S4.py
python paper_fig3.py
python script_paper_Fig4_stats.py
python script_paper_fig5_tableS5.py
python script_paper_Fig5_c_and_d_TableS6.py
python script_paper_Fig6_Table7.py
python script_paper_Fig7_TableS8.py
python paper_FigS2.py
python script_paper_FigS1_regional_afforestation_Stats_low_emission_scenario.py
R10 regions map (Figure S3): `r10_regions_map_from_mask_smithetal.ipynb`
```
