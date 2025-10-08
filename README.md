# Tennis ML

Machine learning workflows for predicting tennis match outcomes from point‑level and odds data. The repo contains notebooks to process data, create a training set, train XGBoost models, and evaluate predictions.

## Requirements

- **Python**: 3.13 (see `pyproject.toml` for the spec)
- **OS packages**: none required for typical installs
- **Optional**: [DVC](https://dvc.org/) if you want to pull data/models from a remote

## Install

Option A — pip/venv

- Create and activate a virtual env, then install project deps defined in `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
# If you use DVC remotes for data/models:
pip install dvc  # or: pip install "dvc[s3]" for S3
```

Option B — uv (uses `uv.lock`)

```bash
uv venv
source .venv/bin/activate
uv sync
```

JupyterLab is included in dependencies, so you can launch notebooks after install.

## Data and Models

This repo uses DVC to track `data/` and `models/` artifacts, but no remote is configured in the repo (`.dvc/config` is empty). You have two paths to get data/models:

- With a configured remote: run `dvc pull` to fetch `data/` and `models/`.
- Without a remote: build everything locally by running the notebooks in the order below.

## Run Order (Notebooks)

1) `data_processing.ipynb`

- Builds per‑match parquet files under `data/matches/` from raw data (feature engineering and cleanup). Make sure any raw inputs expected by the notebook exist under `data/` (e.g., `data/raw data/`).

Outputs:
- `data/matches/*.parquet`

2) `Setup_Data.ipynb`

- Loads all parquet files from `data/matches/` and assembles the model training table.

Outputs:
- `data/processed/data_processed.parquet`

3) Training notebooks

- `model_training.ipynb` — trains an XGBoost model on `data/processed/data_processed.parquet`, saves a model file, and updates the model registry parquet.
- Variants:
  - `model_training_single.ipynb` — trains a variant using a smaller feature set.
  - `model_training no flags.ipynb` — trains a variant without boolean “flag” features.

Outputs:
- `models/xgb_model_*.json`
- `model_data.parquet` (registry of saved models and metadata)

4) `testing model.ipynb`

- Loads a trained model from `model_data.parquet` and evaluates/presents match win probability curves. Requires files from step 3 to exist.

Optional — `polymarket_vis.ipynb`

- Demonstrates pulling market/odds data and visualizing. If using The Odds API, provide a key via environment variable:

```bash
export ODDS_API_KEY="<your_api_key>"
```

Then update the notebook cell that references the API key to read from the environment or paste your key for local experiments.

## How to Run

Launch JupyterLab and open the notebooks in order:

```bash
jupyter lab
```

If you used the training notebooks, you should see:

- `data/matches/*.parquet` and `data/processed/data_processed.parquet`
- trained models under `models/`
- `model_data.parquet` listing available models

## Notes & Tips

- Python version: see `pyproject.toml:5` for `requires-python`. If Python 3.13 is not available on your system, you can try a slightly lower version, but you may need to adjust `pyproject.toml` accordingly.
- XGBoost wheels: if you hit installation issues with `xgboost` on your platform/Python combo, try a recent wheel (e.g., `pip install "xgboost>=2.0"`).
- DVC remotes: if `dvc pull` reports “no remote,” that’s expected without a configured remote. Run the notebooks instead to (re)build artifacts.
- Paths: training notebooks expect `data/processed/data_processed.parquet`. If you previously produced `data/data_processed.parquet`, rerun `Setup_Data.ipynb` to write to the expected location.

## License

MIT
