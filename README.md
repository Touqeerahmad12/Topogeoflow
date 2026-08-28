# TopoGeoFlow

This project implements a reproducible research workflow for evaluating topology-constrained conditional flow matching for minority-class generation in imbalanced learning.

## Setup

```bash
conda env create -f environment.yml
conda activate topogeoflow
```

or with pip:

```bash
pip install -r requirements.txt
```

## Quick start

Run the resampling benchmark (and TopoGeoFlow ablation study) on a dataset:

```bash
python run_all.py --dataset path/to/data.csv --output-dir results/my_run --folds 5 --seed 42
```

All arguments are optional:

| Flag              | Default                          | Description                              |
|-------------------|-----------------------------------|-------------------------------------------|
| `--dataset`        | `../Dataset/Toy_imbalanced.csv`   | Path to the input CSV dataset             |
| `--output-dir`      | `results/Toy_imbalanced`          | Where results/figures are written         |
| `--folds`          | `5`                                | Number of stratified CV folds             |
| `--seed`           | `42`                               | Random seed                                |
| `--skip-ablation`  | off                                | Skip the TopoGeoFlow ablation study        |

Then analyze the resulting tables:

```bash
python analyze_resampling_results.py --data-dir data --output-dir results --alpha 0.05
```

This reads `data/auc.csv`, `data/accuracy.csv`, `data/f1.csv`, and `data/gmean.csv`, and produces summary tables, Friedman/Nemenyi critical-difference diagrams, boxplot grids, and a mean-rank heatmap in `--output-dir`.
