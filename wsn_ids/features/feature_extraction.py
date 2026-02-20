"""
Feature Extraction and Preprocessing for WSN-IDS

Responsibilities:
  - Load raw dataset
  - Validate and clean features
  - Derive secondary features (energy efficiency index, anomaly score proxy)
  - Scale features for distance-based models
  - Train/test split with stratification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from wsn_ids import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Secondary / derived features
# ---------------------------------------------------------------------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional hand-crafted features that capture compound
    WSN anomaly signals.

    New columns added:
      energy_efficiency_index  : forwarding_rate * residual_energy / 100
                                 Low value = either dropping packets or low battery
      traffic_anomaly_score    : drop_ratio * (1 / (tx_interval + 1e-9)) * 1000
                                 High value = many drops at high tx rate → DoS/SF
      identity_confusion_index : neighbor_count * signal_variation
                                 High value → Sybil / Hello-Flood signature
    """
    df = df.copy()

    df["energy_efficiency_index"] = (
        df["packet_forwarding_rate"] * df["residual_energy"] / 100.0
    ).round(4)

    df["traffic_anomaly_score"] = (
        df["packet_drop_ratio"] * (1000.0 / (df["transmission_interval"] + 1e-9))
    ).round(4)

    df["identity_confusion_index"] = (
        df["neighbor_count"] * df["signal_strength_variation"]
    ).round(4)

    return df


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str, add_derived: bool = True) -> pd.DataFrame:
    """Load the WSN dataset CSV and optionally add derived features."""
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")

    if add_derived:
        df = add_derived_features(df)

    return df


# ---------------------------------------------------------------------------
# Train / Test split
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """
    Stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray arrays
    feature_cols                      : list of feature names used
    """
    base = list(FEATURE_NAMES)
    derived = ["energy_efficiency_index", "traffic_anomaly_score", "identity_confusion_index"]
    all_features = base + [c for c in derived if c in df.columns]

    if feature_cols is None:
        feature_cols = all_features

    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test, feature_cols


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

def get_scaler(kind: str = "standard") -> StandardScaler | MinMaxScaler:
    """Return a fitted-ready scaler instance ('standard' or 'minmax')."""
    if kind == "minmax":
        return MinMaxScaler()
    return StandardScaler()


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    kind: str = "standard",
) -> tuple[np.ndarray, np.ndarray, StandardScaler | MinMaxScaler]:
    """
    Fit scaler on training data and transform both splits.

    Returns scaled X_train, scaled X_test, fitted scaler.
    """
    scaler = get_scaler(kind)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ---------------------------------------------------------------------------
# Feature importance summary
# ---------------------------------------------------------------------------

def feature_importance_table(
    importances: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    return (
        pd.DataFrame({"feature": feature_cols, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
