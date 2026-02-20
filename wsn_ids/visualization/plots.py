"""
Visualization Module for WSN-IDS

All functions save figures to `save_dir` and optionally display them.
Figures generated:
  1.  class_distribution.png       – bar chart of attack type counts
  2.  feature_distributions.png    – box plots per feature per class
  3.  correlation_heatmap.png      – Pearson correlation of features
  4.  confusion_matrix_<model>.png – heat-map confusion matrix
  5.  model_comparison.png         – grouped bar chart of all metrics
  6.  feature_importance.png       – horizontal bar (Random Forest)
  7.  roc_curves.png               – one-vs-rest ROC per class (best model)
  8.  pca_scatter.png              – 2-D PCA scatter coloured by class
  9.  energy_impact.png            – inference latency vs accuracy scatter
  10. cv_scores.png                – cross-validation fold scores
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix

from wsn_ids import ATTACK_LABELS

# Consistent colour palette
PALETTE = list(mcolors.TABLEAU_COLORS.values())[:7]
sns.set_theme(style="whitegrid", font_scale=1.1)


def _savefig(fig, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame, save_dir: str = "results") -> None:
    counts = df["attack_type"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(counts.index, counts.values, color=PALETTE, edgecolor="white")
    ax.bar_label(bars, padding=4)
    ax.set_title("WSN Dataset — Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Number of Samples")
    plt.xticks(rotation=25, ha="right")
    _savefig(fig, Path(save_dir) / "class_distribution.png")


# ---------------------------------------------------------------------------
# 2. Feature distributions per class
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    df: pd.DataFrame,
    features: list[str],
    save_dir: str = "results",
) -> None:
    n = len(features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        data = [df[df["label"] == lbl][feat].values for lbl in sorted(ATTACK_LABELS)]
        bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True)
        for patch, colour in zip(bp["boxes"], PALETTE):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(ATTACK_LABELS) + 1))
        ax.set_xticklabels(
            [ATTACK_LABELS[k] for k in sorted(ATTACK_LABELS)],
            rotation=30, ha="right", fontsize=8,
        )
        ax.set_title(feat.replace("_", " ").title(), fontsize=11)
        ax.set_ylabel("Value")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions by Attack Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, Path(save_dir) / "feature_distributions.png")


# ---------------------------------------------------------------------------
# 3. Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: list[str],
    save_dir: str = "results",
) -> None:
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        square=True, linewidths=0.5, ax=ax,
        xticklabels=[f.replace("_", "\n") for f in features],
        yticklabels=[f.replace("_", "\n") for f in features],
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, Path(save_dir) / "correlation_heatmap.png")


# ---------------------------------------------------------------------------
# 4. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: str = "results",
) -> None:
    labels = sorted(np.unique(y_true))
    class_names = [ATTACK_LABELS[l] for l in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_norm, annot=cm, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    _savefig(fig, Path(save_dir) / f"confusion_matrix_{slug}.png")


# ---------------------------------------------------------------------------
# 5. Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    summary_df: pd.DataFrame,
    save_dir: str = "results",
) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    models = summary_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, metric in enumerate(metrics):
        vals = summary_df[metric].values
        bars = ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").title(),
                      color=PALETTE[i], alpha=0.88, edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5, padding=2)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — WSN Intrusion Detection", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _savefig(fig, Path(save_dir) / "model_comparison.png")


# ---------------------------------------------------------------------------
# 6. Feature importance (tree-based models)
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    model_name: str = "Random Forest",
    save_dir: str = "results",
) -> None:
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [feature_names[i].replace("_", " ").title() for i in idx],
        importances[idx],
        color=PALETTE[0], edgecolor="white",
    )
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, Path(save_dir) / "feature_importance.png")


# ---------------------------------------------------------------------------
# 7. ROC curves (one-vs-rest, best model)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    save_dir: str = "results",
) -> None:
    classes = sorted(np.unique(y_test))
    y_bin = label_binarize(y_test, classes=classes)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        print(f"  [ROC] {model_name} has no probability output, skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, cls in enumerate(classes):
        if y_score.ndim == 1:
            break
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                label=f"{ATTACK_LABELS[cls]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    _savefig(fig, Path(save_dir) / "roc_curves.png")


# ---------------------------------------------------------------------------
# 8. PCA 2-D scatter
# ---------------------------------------------------------------------------

def plot_pca_scatter(
    X: np.ndarray,
    y: np.ndarray,
    save_dir: str = "results",
    n_samples: int = 2000,
) -> None:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_s, y_s = X[idx], y[idx]

    pca = PCA(n_components=2, random_state=0)
    X_pca = pca.fit_transform(X_s)

    fig, ax = plt.subplots(figsize=(10, 7))
    for lbl, colour in zip(sorted(ATTACK_LABELS), PALETTE):
        mask = y_s == lbl
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colour, label=ATTACK_LABELS[lbl],
            alpha=0.6, s=22, edgecolors="none",
        )
    ax.set_title("PCA — 2D Projection of WSN Feature Space", fontsize=13, fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(markerscale=1.5, fontsize=9)
    plt.tight_layout()
    _savefig(fig, Path(save_dir) / "pca_scatter.png")


# ---------------------------------------------------------------------------
# 9. Energy impact vs accuracy scatter
# ---------------------------------------------------------------------------

def plot_energy_impact(summary_df: pd.DataFrame, save_dir: str = "results") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, row in summary_df.iterrows():
        ax.scatter(row["inference_ms"], row["accuracy"] * 100,
                   s=120, color=PALETTE[i % len(PALETTE)], zorder=3)
        ax.annotate(row["model"], (row["inference_ms"], row["accuracy"] * 100),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Inference Time (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Energy Impact vs Accuracy Trade-off", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, Path(save_dir) / "energy_impact.png")


# ---------------------------------------------------------------------------
# 10. Cross-validation scores
# ---------------------------------------------------------------------------

def plot_cv_scores(cv_df: pd.DataFrame, model_name: str, save_dir: str = "results") -> None:
    df = cv_df.drop(index=["mean", "std"], errors="ignore").copy()
    df = df[df.index != "mean"]
    metrics = ["accuracy", "f1", "precision", "recall"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        if metric in df.columns:
            ax.plot(df["fold"], df[metric], marker="o",
                    label=metric.title(), color=PALETTE[i])

    ax.set_ylim(0.7, 1.05)
    ax.set_xticks(df["fold"])
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_title(f"Cross-Validation Scores — {model_name}", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, Path(save_dir) / "cv_scores.png")
