This repository is linked to the following research paper prepared for peer-review: Sirisha Kalidindi, Gabriel Abrahão, Anna B. Harper, Michael Crawford, Jens Heinke, Elmar Kriegler, Joeri Rogelj (2026). Quantifying uncertainty in land-based carbon removals in AR6 mitigation pathways.

The repository includes the Python scripts to reproduce Figures and the analysis in the paper. 
## Install from GitHub

To clone this repository, use the following command:

```bash
git clone git@github.com:sirishakdr/LandCDR-Uncertainty-ESM2025.git
cd LandCDR-Uncertainty-ESM2025
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate your_env_name
```

## Run Python Scripts

Run the Python scripts in the folder from the terminal with:

```bash
python filename.py
```

## Data Requirements

Datasets used in the study and required by the Python scripts are available from the following sources:

- [Zenodo](https://zenodo.org/uploads/20309536)
- IIASA Scenario Database:
  - [AR6 Scenarios Database metadata indicators v1.1](https://data.ece.iiasa.ac.at/ar6/#/downloads/AR6_Scenarios_Database_metadata_indicators_v1.1)
  - [AR6 Scenarios Database World v1.1](https://data.ece.iiasa.ac.at/ar6/#/downloads/AR6_Scenarios_Database_World_v1.1)
  - [AR6 Scenarios Database R10 regions v1.1](https://data.ece.iiasa.ac.at/ar6/#/downloads/AR6_Scenarios_Database_R10_regions_v1.1)
- [R10 regions mask](https://doi.org/10.5281/zenodo.8362562)

Download all datasets listed above into the same parent directory as the cloned Git repository before running the scripts locally.
