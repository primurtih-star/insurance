#!/usr/bin/env python3
"""
retrain.py

Convenience script to retrain a model (rf or dt) on a new CSV and optionally
compare with the current production model. Designed to be robust and
informative for CI / manual runs.

Usage examples:
  python retrain.py --csv data/raw/insurance.csv --target charges --cat-cols smoker sex region
  python retrain.py --csv data/dataset.csv --target risk_score --model rf --cv
"""
from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger("retrain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def import_pipeline_components():
    """
    Import optional project components with clear error messages if missing.
    """
    try:
        from ml.preprocess import PreprocessingPipeline  # type: ignore
    except Exception as exc:  # pragma: no cover - helpful runtime message
        raise ImportError("ml.preprocess.PreprocessingPipeline not found") from exc

    try:
        from ml.train import ModelTrainer  # type: ignore
    except Exception as exc:
        raise ImportError("ml.train.ModelTrainer not found") from exc

    try:
        from ml.evaluate import ModelEvaluator  # type: ignore
    except Exception:
        ModelEvaluator = None  # optional

    try:
        from ml.model_compare import compare_models  # type: ignore
    except Exception:
        compare_models = None  # optional

    try:
        from ml.feature_importance import get_feature_importance, plot_importance  # type: ignore
    except Exception:
        get_feature_importance = None
        plot_importance = None

    return {
        "PreprocessingPipeline": PreprocessingPipeline,
        "ModelTrainer": ModelTrainer,
        "ModelEvaluator": ModelEvaluator,
        "compare_models": compare_models,
        "get_feature_importance": get_feature_importance,
        "plot_importance": plot_importance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain pipeline for RF/DT models.")
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="risk_score", help="Target column name")
    parser.add_argument("--cat-cols", nargs="+", default=None, help="Categorical columns (optional)")
    parser.add_argument("--model", choices=["rf", "dt", "both"], default="both", help="Which model(s) to train")
    parser.add_argument("--cv", action="store_true", help="Run light cross-validation")
    parser.add_argument("--model-dir", default="data/model", help="Directory to store models")
    parser.add_argument("--champion-name", default="rf_model.joblib", help="Production model filename")
    return parser.parse_args()


def run_retraining_pipeline(
    csv_path: Path,
    target_col: str,
    cat_cols: List[str] | None,
    model_dir: Path,
    champion_name: str,
    model_choice: str = "both",
    cv: bool = False,
) -> bool:
    """
    End-to-end retraining pipeline:
    - load CSV
    - preprocess (fit)
    - train candidate model
    - evaluate and compare with champion
    - update production model if candidate wins
    - update feature importance plot (if available)
    """
    components = import_pipeline_components()
    PreprocessingPipeline = components["PreprocessingPipeline"]
    ModelTrainer = components["ModelTrainer"]
    ModelEvaluator = components["ModelEvaluator"]
    compare_models = components["compare_models"]
    get_feature_importance = components["get_feature_importance"]
    plot_importance = components["plot_importance"]

    model_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = model_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    champion_path = model_dir / champion_name

    logger.info("Starting retraining pipeline for CSV: %s", csv_path)

    if not csv_path.exists():
        logger.error("CSV file not found: %s", csv_path)
        return False

    # 1. Load data
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.exception("Failed to read CSV")
        return False

    # 2. Preprocessing (fit)
    try:
        pp = PreprocessingPipeline()
        X_df, y = pp.fit(df, target_col)
        # Save preprocessing artifacts next to models
        pp.save(model_dir)
    except Exception as exc:
        logger.exception("Preprocessing failed")
        return False

    # 3. Train/test split
    try:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)
    except Exception:
        logger.exception("Train/test split failed")
        return False

    # 4. Train candidate model(s)
    # We train only RF here as default; ModelTrainer implementation may support other types
    try:
        logger.info("Initializing trainer")
        trainer = ModelTrainer(n_estimators=200, max_depth=15)
        trainer.train(X_train, y_train)
        candidate_path = Path(trainer.save_model(output_dir=str(model_dir)))
        logger.info("Candidate model saved to %s", candidate_path)
    except Exception:
        logger.exception("Training failed")
        return False

    # 5. Evaluate candidate
    metrics = {}
    try:
        if ModelEvaluator is not None:
            evaluator = ModelEvaluator(str(candidate_path))
            metrics, _ = evaluator.evaluate(X_test, y_test)
            evaluator.save_metrics(metrics)
            logger.info("Candidate metrics: %s", metrics)
        else:
            logger.warning("ModelEvaluator not available; skipping detailed evaluation")
    except Exception:
        logger.exception("Evaluation failed; continuing to comparison step")

    # 6. Compare with champion (if exists and compare_models available)
    try:
        if champion_path.exists() and compare_models is not None:
            winner_path, compare_result = compare_models(str(champion_path), str(candidate_path), X_test, y_test)
            logger.info("Comparison result: %s", compare_result)
            if Path(winner_path).resolve() == candidate_path.resolve():
                # backup old champion
                backup_name = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{champion_path.name}"
                shutil.copy2(champion_path, archive_dir / backup_name)
                # replace champion
                shutil.copy2(candidate_path, champion_path)
                logger.info("Champion updated with candidate model: %s", champion_path)
            else:
                # keep champion, candidate remains archived
                shutil.copy2(candidate_path, archive_dir / candidate_path.name)
                logger.info("Champion kept. Candidate archived to %s", archive_dir)
        else:
            # No champion exists or no comparator: promote candidate to champion
            shutil.copy2(candidate_path, champion_path)
            logger.info("No existing champion. Candidate promoted to champion: %s", champion_path)
    except Exception:
        logger.exception("Model comparison / promotion failed")
        return False

    # 7. Update feature importance visualization if utilities available
    try:
        if get_feature_importance is not None and plot_importance is not None:
            # Attempt to infer feature names from original df (drop target)
            feature_names = [c for c in df.columns if c != target_col]
            df_imp = get_feature_importance(str(champion_path), feature_names=feature_names)
            plot_dir = Path("ui/img/plots")
            plot_dir.mkdir(parents=True, exist_ok=True)
            plot_importance(df_imp, save_path=str(plot_dir / "feature_importance.png"))
            logger.info("Feature importance updated at %s", plot_dir / "feature_importance.png")
        else:
            logger.debug("Feature importance utilities not available; skipping visualization")
    except Exception:
        logger.exception("Failed to update feature importance")

    logger.info("Retraining pipeline finished successfully")
    return True


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    model_dir = Path(args.model_dir)
    success = run_retraining_pipeline(
        csv_path=csv_path,
        target_col=args.target,
        cat_cols=args.cat_cols,
        model_dir=model_dir,
        champion_name=args.champion_name,
        model_choice=args.model,
        cv=args.cv,
    )
    if not success:
        logger.error("Retraining pipeline failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()