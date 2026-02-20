"""
Model Evaluation for WSN-IDS

Computes per-model and per-class metrics:
  - Accuracy
  - Precision (macro)
  - Recall    (macro)
  - F1-Score  (macro)
  - False Positive Rate (macro)
  - Energy Consumption Impact score (proxy based on inference cost)

Also provides a summary DataFrame and cross-validation helper.
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from wsn_ids import ATTACK_LABELS


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Macro-averaged False Positive Rate across all classes.

    FPR_c = FP_c / (FP_c + TN_c)
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    fprs = []
    for c in range(n_classes):
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    return float(np.mean(fprs))


def energy_impact_score(inference_time_ms: float) -> float:
    """
    Proxy energy consumption impact score.

    Maps inference latency to a 0–1 scale where lower is better.
    Assumes 500 ms as an upper bound for a resource-constrained node.
    """
    return round(1.0 - min(inference_time_ms / 500.0, 1.0), 4)


def evaluate_model(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate a single fitted model and return a metrics dict.
    """
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    fpr = false_positive_rate(y_test, y_pred)
    energy = energy_impact_score(elapsed_ms)

    return {
        "model": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "energy_impact": energy,
        "inference_ms": round(elapsed_ms, 2),
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    models: dict[str, object],
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """
    Evaluate all models and return:
      - summary_df : DataFrame with one row per model
      - details    : dict {model_name: full metrics dict incl. y_pred}
    """
    rows = []
    details = {}

    for name, model in models.items():
        result = evaluate_model(name, model, X_test, y_test)
        details[name] = result
        rows.append({k: v for k, v in result.items() if k != "y_pred"})
        if verbose:
            print(
                f"  {name:<20} "
                f"Acc={result['accuracy']:.4f}  "
                f"F1={result['f1_score']:.4f}  "
                f"FPR={result['false_positive_rate']:.4f}  "
                f"EI={result['energy_impact']:.4f}"
            )

    summary_df = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    return summary_df, details


# ---------------------------------------------------------------------------
# Per-class report
# ---------------------------------------------------------------------------

def per_class_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Return sklearn classification_report as a DataFrame."""
    labels = sorted(np.unique(y_test))
    target_names = [ATTACK_LABELS[l] for l in labels]
    report = classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    df = pd.DataFrame(report).T
    return df


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run stratified k-fold cross-validation and return fold-level results.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    df = pd.DataFrame({
        "fold": range(1, n_splits + 1),
        "accuracy": results["test_accuracy"],
        "f1": results["test_f1_macro"],
        "precision": results["test_precision_macro"],
        "recall": results["test_recall_macro"],
    })
    df.loc["mean"] = df.mean()
    df.loc["std"] = df.std()
    return df.round(4)
