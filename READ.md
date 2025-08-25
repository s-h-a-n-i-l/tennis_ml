# Tennis ML

This repository explores machine learning approaches for predicting tennis match outcomes using sports odds data.

## Project Structure

- `data_processing.ipynb` – notebook for fetching and preparing raw odds data.
- `model_training.ipynb` – trains XGBoost and scikit-learn models.
- `polymarket_vis.ipynb` – example of pulling market data and visualizing it.
- `data/` and `models/` – large datasets and trained models tracked with [DVC](https://dvc.org/).

## Installation

This project targets **Python 3.13** and uses a `pyproject.toml` for dependencies. Install requirements with [pip](https://pip.pypa.io/):

```bash
pip install -e .
```

## Data and Model Access

Data files and trained models are stored via DVC. Fetch them before running notebooks:

```bash
dvc pull
```

## Usage

Launch JupyterLab to run and modify the notebooks:

```bash
jupyter lab
```

The notebooks demonstrate how to gather sports betting data, engineer features, and train predictive models for tennis outcomes.

## License

This project is released under the MIT License.