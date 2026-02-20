"""
WSN Intrusion Detection System — Core IDS Module

Provides:
  - WSNIDS class  : end-to-end pipeline (generate → train → evaluate → alert)
  - predict_node  : classify a single node observation
  - alert         : human-readable alert message

Usage
-----
    from wsn_ids.ids import WSNIDS

    ids = WSNIDS()
    ids.run()                          # full pipeline
    label, name = ids.predict_node(observation)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from wsn_ids import ATTACK_LABELS, FEATURE_NAMES
from wsn_ids.data.generate_dataset import generate_dataset
from wsn_ids.features.feature_extraction import (
    add_derived_features,
    split_dataset,
)
from wsn_ids.models.train import build_models, train_all, save_models
from wsn_ids.models.evaluate import evaluate_all, per_class_report, cross_validate_model
from wsn_ids.visualization.plots import (
    plot_class_distribution,
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_importance,
    plot_roc_curves,
    plot_pca_scatter,
    plot_energy_impact,
    plot_cv_scores,
)


class WSNIDS:
    """
    End-to-end WSN Intrusion Detection System pipeline.

    Parameters
    ----------
    samples_per_class : int    – dataset size per attack type
    results_dir       : str    – where to write models, CSVs, PNGs
    random_state      : int    – global RNG seed
    """

    def __init__(
        self,
        samples_per_class: int = 500,
        results_dir: str = "results",
        random_state: int = 42,
    ):
        self.samples_per_class = samples_per_class
        self.results_dir = Path(results_dir)
        self.random_state = random_state

        self.df: pd.DataFrame | None = None
        self.models: dict | None = None
        self.summary: pd.DataFrame | None = None
        self.details: dict | None = None
        self.feature_cols: list[str] | None = None
        self.best_model_name: str | None = None

    # ------------------------------------------------------------------
    # Step 1 — Data
    # ------------------------------------------------------------------

    def generate_data(self) -> "WSNIDS":
        print("\n[1/5] Generating WSN dataset ...")
        csv_path = self.results_dir / "wsn_dataset.csv"
        self.df = generate_dataset(
            samples_per_class=self.samples_per_class,
            seed=self.random_state,
            save_path=str(csv_path),
        )
        self.df = add_derived_features(self.df)
        print(f"      Total samples : {len(self.df)}")
        print(f"      Classes       : {self.df['attack_type'].nunique()}")
        return self

    # ------------------------------------------------------------------
    # Step 2 — Visualise dataset
    # ------------------------------------------------------------------

    def visualise_data(self) -> "WSNIDS":
        print("\n[2/5] Generating exploratory visualisations ...")
        viz_dir = self.results_dir / "plots"
        viz_dir.mkdir(parents=True, exist_ok=True)

        all_features = list(FEATURE_NAMES) + [
            "energy_efficiency_index",
            "traffic_anomaly_score",
            "identity_confusion_index",
        ]
        available = [f for f in all_features if f in self.df.columns]

        plot_class_distribution(self.df, save_dir=str(viz_dir))
        plot_feature_distributions(self.df, features=available, save_dir=str(viz_dir))
        plot_correlation_heatmap(self.df, features=available, save_dir=str(viz_dir))

        X_all = self.df[available].values
        y_all = self.df["label"].values
        plot_pca_scatter(X_all, y_all, save_dir=str(viz_dir))
        return self

    # ------------------------------------------------------------------
    # Step 3 — Train
    # ------------------------------------------------------------------

    def train(self) -> "WSNIDS":
        print("\n[3/5] Training models ...")
        X_train, X_test, y_train, y_test, self.feature_cols = split_dataset(
            self.df, test_size=0.20, random_state=self.random_state
        )
        self._X_train = X_train
        self._X_test = X_test
        self._y_train = y_train
        self._y_test = y_test

        self.models = build_models(random_state=self.random_state)
        train_all(self.models, X_train, y_train)
        save_models(self.models, save_dir=str(self.results_dir / "models"))
        return self

    # ------------------------------------------------------------------
    # Step 4 — Evaluate
    # ------------------------------------------------------------------

    def evaluate(self) -> "WSNIDS":
        print("\n[4/5] Evaluating models ...")
        self.summary, self.details = evaluate_all(
            self.models, self._X_test, self._y_test
        )

        self.best_model_name = self.summary.iloc[0]["model"]
        print(f"\n  Best model: {self.best_model_name}")

        # Save summary CSV
        summary_path = self.results_dir / "model_summary.csv"
        self.summary.to_csv(summary_path, index=False)
        print(f"  Saved: {summary_path}")

        # Per-class report for best model
        best_pred = self.details[self.best_model_name]["y_pred"]
        report_df = per_class_report(self._y_test, best_pred)
        report_path = self.results_dir / "per_class_report.csv"
        report_df.to_csv(report_path)
        print(f"  Saved: {report_path}")

        # Cross-validation on best model
        print(f"\n  Running 5-fold CV on {self.best_model_name} ...")
        best_model = self.models[self.best_model_name]
        X_all = np.vstack([self._X_train, self._X_test])
        y_all = np.concatenate([self._y_train, self._y_test])
        self._cv_df = cross_validate_model(best_model, X_all, y_all)
        cv_path = self.results_dir / "cv_results.csv"
        self._cv_df.to_csv(cv_path)
        print(f"  Saved: {cv_path}")
        return self

    # ------------------------------------------------------------------
    # Step 5 — Visualise results
    # ------------------------------------------------------------------

    def visualise_results(self) -> "WSNIDS":
        print("\n[5/5] Generating result visualisations ...")
        viz_dir = self.results_dir / "plots"
        viz_dir.mkdir(parents=True, exist_ok=True)

        plot_model_comparison(self.summary, save_dir=str(viz_dir))
        plot_energy_impact(self.summary, save_dir=str(viz_dir))
        plot_cv_scores(self._cv_df, self.best_model_name, save_dir=str(viz_dir))

        best_model = self.models[self.best_model_name]
        best_pred = self.details[self.best_model_name]["y_pred"]

        plot_confusion_matrix(
            self._y_test, best_pred,
            model_name=self.best_model_name,
            save_dir=str(viz_dir),
        )

        # ROC for best model
        plot_roc_curves(
            best_model, self._X_test, self._y_test,
            model_name=self.best_model_name,
            save_dir=str(viz_dir),
        )

        # Feature importance (tree-based)
        rf_name = "Random Forest"
        if rf_name in self.models:
            rf = self.models[rf_name]
            estimator = rf if hasattr(rf, "feature_importances_") else (
                rf.named_steps.get("rf") or
                next((v for v in rf.named_steps.values()
                      if hasattr(v, "feature_importances_")), None)
            )
            if estimator is not None:
                plot_feature_importance(
                    estimator.feature_importances_,
                    self.feature_cols,
                    model_name=rf_name,
                    save_dir=str(viz_dir),
                )

        return self

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> "WSNIDS":
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.generate_data()
        self.visualise_data()
        self.train()
        self.evaluate()
        self.visualise_results()
        self._print_summary()
        return self

    def _print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("  WSN-IDS  —  Final Results Summary")
        print("=" * 60)
        cols = ["model", "accuracy", "f1_score", "false_positive_rate", "energy_impact"]
        print(self.summary[cols].to_string(index=False))
        print("=" * 60)
        print(f"\n  Best Model     : {self.best_model_name}")
        best = self.summary.iloc[0]
        print(f"  Accuracy       : {best['accuracy']:.4f}")
        print(f"  F1-Score       : {best['f1_score']:.4f}")
        print(f"  FPR            : {best['false_positive_rate']:.4f}")
        print(f"  Energy Impact  : {best['energy_impact']:.4f}")
        print(f"\n  All outputs saved in: {self.results_dir.resolve()}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    def predict_node(
        self,
        observation: list[float] | np.ndarray,
    ) -> tuple[int, str]:
        """
        Classify a single node observation using the best trained model.

        Parameters
        ----------
        observation : list or array of length == len(feature_cols)

        Returns
        -------
        (label_int, attack_name_str)
        """
        if self.models is None or self.best_model_name is None:
            raise RuntimeError("Run ids.run() or ids.train() before calling predict_node().")

        x = np.array(observation, dtype=float).reshape(1, -1)

        # If derived features are expected, we derive them on-the-fly
        if len(x[0]) == len(FEATURE_NAMES):
            row = pd.DataFrame([observation], columns=FEATURE_NAMES)
            from wsn_ids.features.feature_extraction import add_derived_features
            row = add_derived_features(row)
            x = row[self.feature_cols].values

        model = self.models[self.best_model_name]
        label = int(model.predict(x)[0])
        return label, ATTACK_LABELS[label]

    def alert(self, observation: list[float] | np.ndarray) -> str:
        """Return a formatted alert string for a node observation."""
        label, name = self.predict_node(observation)
        if label == 0:
            return f"[OK]    Node behaviour is NORMAL."
        return (
            f"[ALERT] Intrusion detected!\n"
            f"        Attack Type : {name} (label={label})\n"
            f"        Action      : Isolate node and notify base station."
        )
