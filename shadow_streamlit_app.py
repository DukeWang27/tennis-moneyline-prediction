"""Run the normal Streamlit app with isolated candidate artifacts."""

from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import tennis_predictor.ui.pages as pages


# Save the project's original path factory before replacing it.
_original_paths = pages._paths


_SHADOW_OVERRIDES = {
    "model": Path(
        "outputs/shadow/candidate_live.joblib"
    ),
    "metrics": Path(
        "outputs/candidate-cleaned-kaggle-2020/"
        "metrics.csv"
    ),
    "predictions": Path(
        "outputs/candidate-cleaned-kaggle-2020/"
        "predictions.parquet"
    ),
    "calibration": Path(
        "outputs/candidate-cleaned-kaggle-2020/"
        "calibration_curves.csv"
    ),
    "features": Path(
        "data/processed/features_full.parquet"
    ),
    "matches": Path(
        "data/shadow/candidate_matches.parquet"
    ),
    "processed": Path(
        "data/shadow/candidate_matches.parquet"
    ),
}


def shadow_paths():
    """Start with valid project paths, then replace candidate artifacts."""
    base = _original_paths()

    if not is_dataclass(base):
        raise TypeError(
            "DashboardPaths is expected to be a dataclass, "
            f"but received {type(base)!r}."
        )

    available_fields = {
        field.name
        for field in fields(base)
    }

    updates = {
        name: path
        for name, path in _SHADOW_OVERRIDES.items()
        if name in available_fields
    }

    required = {
        "model",
        "metrics",
        "predictions",
    }

    missing = required - set(updates)

    if missing:
        raise RuntimeError(
            "Could not override required dashboard paths: "
            + ", ".join(sorted(missing))
            + ". Available fields: "
            + ", ".join(sorted(available_fields))
        )

    return replace(base, **updates)


pages._paths = shadow_paths


# Execute the project's existing dashboard entry point.
app_path = Path("streamlit_app.py")

code = compile(
    app_path.read_text(),
    str(app_path),
    "exec",
)

exec(code, globals(), globals())
