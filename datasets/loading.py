"""Strict local CSV loading; no fitting, downloading, deduplication or sorting."""
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DatasetConfig, CAR, ENERGY, student_config

class DatasetSchemaError(ValueError):
    """The supplied CSV differs from the documented original schema."""

@dataclass
class LoadedDataset:
    config: DatasetConfig
    X: pd.DataFrame
    y: pd.Series
    temporal_metadata: pd.DataFrame
    raw_columns: tuple[str, ...]

    @property
    def original_feature_count(self):
        return self.X.shape[1]


def load_dataset(config: DatasetConfig, path=None):
    source = Path(path) if path is not None else config.path
    if not source.is_file():
        raise FileNotFoundError(f"Missing {config.display_name} CSV: {source}. See datasets/README.md for acquisition instructions; no substitute is generated.")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter=config.separator), [])
    missing = sorted(set(config.raw_columns) - set(header))
    extra = sorted(set(header) - set(config.raw_columns))
    if missing or extra or len(header) != len(set(header)):
        raise DatasetSchemaError(f"{source.name}: expected {len(config.raw_columns)} columns; got {len(header)}; missing={missing}, unexpected={extra}; duplicate headers are invalid.")
    frame = pd.read_csv(source, sep=config.separator, encoding="utf-8-sig", na_values=["?", ""], keep_default_na=True)
    if len(frame) != config.expected_rows:
        raise DatasetSchemaError(f"{source.name}: expected {config.expected_rows} rows, got {len(frame)}. Wrong subject/version or incomplete file.")
    # Validate even excluded numerical columns (prior grades and car_ID).
    numeric = [c for c in config.raw_columns if c not in (*config.categorical_columns, "date")]
    for column in numeric:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise DatasetSchemaError(f"{source.name}: {column} must be numeric") from exc
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise DatasetSchemaError(f"{source.name}: {column} contains infinity")
    if config.dataset_id == "car_price" and (frame["car_ID"].isna().any() or frame["car_ID"].duplicated().any()):
        raise DatasetSchemaError("car_ID must contain unique nonmissing identifiers")
    if frame[config.target].isna().any():
        raise DatasetSchemaError(f"{source.name}: target {config.target} contains missing values")
    for column in config.categorical_columns:
        # Object dtype with np.nan is compatible with sklearn's imputers.
        frame[column] = frame[column].astype(object).where(frame[column].notna(), np.nan)
    temporal = pd.DataFrame(index=frame.index)
    if "date" in config.raw_columns:
        try:
            dates = pd.to_datetime(frame["date"], format="%Y-%m-%d %H:%M:%S", errors="raise")
        except (ValueError, TypeError) as exc:
            raise DatasetSchemaError("date must contain valid timestamps") from exc
        if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
            raise DatasetSchemaError("date must be present, unique and strictly chronological; input is never silently sorted")
        temporal["date"] = dates
    X = frame.loc[:, config.feature_names].copy()
    if not 14 <= X.shape[1] <= 39:
        raise DatasetSchemaError("Original predictor count must be within 14-39")
    return LoadedDataset(config, X, frame[config.target].copy(), temporal, tuple(header))


def load_student_performance(path=None, *, include_prior_grades: bool = False):
    """Official early-30 profile; full historical grades remain an explicit opt-in."""
    return load_dataset(student_config(include_prior_grades=include_prior_grades), path)


def load_car_price(path=None):
    return load_dataset(CAR, path)


def load_appliances_energy(path=None):
    return load_dataset(ENERGY, path)
