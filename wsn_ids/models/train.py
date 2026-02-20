"""
Model Training for WSN-IDS

Trains and returns five classifiers:
  1. Random Forest          (primary / best performer expected)
  2. Decision Tree          (interpretable baseline)
  3. K-Nearest Neighbours   (distance-based, requires scaling)
  4. Support Vector Machine (kernel trick, requires scaling)
  5. MLP Neural Network     (deep learner, requires scaling)

All models are wrapped in a named dict so callers can iterate uniformly.
"""

import joblib
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models(random_state: int = 42) -> dict[str, object]:
    """
    Return a dict of {name: estimator}.

    KNN, SVM, and MLP are wrapped in a Pipeline with StandardScaler so they
    can be trained on raw (unscaled) data just like the tree-based models.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=3,
            random_state=random_state,
        ),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                decision_function_shape="ovr",
                random_state=random_state,
            )),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=random_state,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_all(
    models: dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> dict[str, object]:
    """
    Fit every model on the training data.

    Returns the same dict with fitted estimators in-place.
    """
    for name, model in models.items():
        if verbose:
            print(f"  Training {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        if verbose:
            print("done")
    return models


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_models(
    models: dict[str, object],
    save_dir: str = "results/models",
) -> None:
    """Persist trained models to disk using joblib."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        path = Path(save_dir) / filename
        joblib.dump(model, path)
        print(f"  Saved: {path}")


def load_models(
    save_dir: str = "results/models",
) -> dict[str, object]:
    """Load all .pkl model files from a directory."""
    models = {}
    for pkl in sorted(Path(save_dir).glob("*.pkl")):
        name = pkl.stem.replace("_", " ").title()
        models[name] = joblib.load(pkl)
    return models
