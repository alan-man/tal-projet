# TAL Project

## Context
Code and notebooks for the TAL UM4IN813 project.

## Collaborators
- Chu Amélie
- Tambellini Alan

---

## Environment & Setup

### Requirements
- Python >= 3.12

### Install dependencies
```bash
uv sync
```

---

## Repository Arborescence

```text
tal-projet/
├── Dataset/                # Training data
├── Test_set/               # Test data
├── Test_submission/        # Submission files (.csv)
├── python/                 # Python scripts used to run experiments (SSH/local)
├── notebooks/
│   ├── eda_plots/          # EDA and plotting notebooks
│   ├── movies/             # Notebooks related to the movies dataset
│   ├── old_notebooks/      # Archived notebooks
│   └── pres/               # Notebooks related to the presidents dataset
├── images/                 # Generated figures and plots
├── npy/                    # Saved arrays and intermediate data (.npy/.csv)
├── results_json/           # Experiment outputs and metrics (.json)
├── pyproject.toml
├── uv.lock
└── README.md
```
