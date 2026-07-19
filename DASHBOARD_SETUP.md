# Streamlit Dashboard Setup

These steps assume your trained files already exist under `outputs/backtest/`.

## Update an existing project folder

1. Download `tennis-streamlit-ui-addon.zip`.
2. In Finder, unzip it.
3. Copy all files and folders from the unzipped add-on into your existing `tennis-moneyline-prediction` folder.
4. Choose **Replace** or **Merge** when macOS asks. This update does not contain `data/`, `outputs/`, `.venv/`, or `config.yaml`, so your trained artifacts remain intact.

## Install the new dashboard dependencies

Open the project in VS Code, then run this in the integrated terminal:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Launch the app

```bash
streamlit run streamlit_app.py
```

Streamlit should open your browser at:

```text
http://localhost:8501
```

Keep the VS Code terminal running while you use the dashboard. Stop it with `Control + C`.

## Expected local artifacts

The app automatically looks for:

```text
outputs/backtest/trained_system.joblib
outputs/backtest/metrics.csv
outputs/backtest/predictions.parquet
```

You can change the paths from the **Artifact paths** section in the app sidebar.

## Prepare for deployment later

```bash
tennis-prepare-dashboard \
  --source outputs/backtest \
  --destination dashboard_artifacts
```

This creates a compact set of files for a future Streamlit Community Cloud deployment without copying the raw tennis CSV archive.
