# ml/preprocess.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Pastikan path import ini benar sesuai struktur foldermu
from ml.utils.cleaner import basic_clean
from ml.utils.validator import validate_dataset
from ml.utils.encoding import detect_column_types

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class PreprocessingPipeline:
    """
    Single entrypoint for fit/transform preprocessing.
    """

    def __init__(self) -> None:
        self.pipeline: ColumnTransformer | None = None
        self.numeric_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.encoded_feature_names: List[str] = []

    def fit(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        validate_dataset(df, target_col)
        df_clean = basic_clean(df.copy())

        y = df_clean[target_col].copy()
        X = df_clean.drop(columns=[target_col])

        numeric_cols, cat_cols = detect_column_types(X)
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols

        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore")), # Update: sparse -> sparse_output untuk sklearn terbaru
            ]
        )

        transformers = []
        if self.numeric_cols:
            transformers.append(("num", numeric_pipeline, self.numeric_cols))
        if self.cat_cols:
            transformers.append(("cat", categorical_pipeline, self.cat_cols))

        if not transformers:
            raise ValueError("No features detected for preprocessing")

        self.pipeline = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)

        logger.info("Fitting preprocessing pipeline (num=%d, cat=%d)", len(self.numeric_cols), len(self.cat_cols))
        X_transformed = self.pipeline.fit_transform(X)

        encoded_names: List[str] = []
        if self.cat_cols:
            try:
                for name, trans, cols in self.pipeline.transformers_:
                    if name == "num":
                        encoded_names.extend(cols)
                    elif name == "cat":
                        ohe = trans.named_steps["onehot"]
                        ohe_names = list(ohe.get_feature_names_out(cols))
                        encoded_names.extend(ohe_names)
            except Exception:
                encoded_names = [f"f{i}" for i in range(X_transformed.shape[1])]
        else:
            encoded_names = [f"f{i}" for i in range(X_transformed.shape[1])]

        self.encoded_feature_names = encoded_names
        X_df = pd.DataFrame(X_transformed, columns=self.encoded_feature_names, index=X.index)

        logger.info("Preprocessing fit complete. Output shape: %s", X_df.shape)
        return X_df, y

    def transform(self, df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted. Call fit(...) first.")

        df_clean = basic_clean(df.copy())
        if target_col and target_col in df_clean.columns:
            X = df_clean.drop(columns=[target_col])
        else:
            X = df_clean

        X_transformed = self.pipeline.transform(X)
        X_df = pd.DataFrame(X_transformed, columns=self.encoded_feature_names, index=X.index)
        return X_df

    def save(self, output_dir: str | Path = "ml/artifacts") -> Dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pipeline_path = out / "preprocessing_pipeline.joblib"
        meta_path = out / "preprocessing_meta.joblib"

        joblib.dump(self.pipeline, pipeline_path)
        joblib.dump(
            {
                "numeric_cols": self.numeric_cols,
                "cat_cols": self.cat_cols,
                "encoded_feature_names": self.encoded_feature_names,
            },
            meta_path,
        )
        logger.info("Saved preprocessing pipeline to %s", pipeline_path)
        return {"pipeline": str(pipeline_path), "meta": str(meta_path)}

    def load(self, pipeline_path: str | Path, meta_path: str | Path) -> None:
        self.pipeline = joblib.load(pipeline_path)
        meta = joblib.load(meta_path)
        self.numeric_cols = meta.get("numeric_cols", [])
        self.cat_cols = meta.get("cat_cols", [])
        self.encoded_feature_names = meta.get("encoded_feature_names", [])
        logger.info("Loaded preprocessing pipeline from %s and %s", pipeline_path, meta_path)

# --- FUNGSI TAMBAHAN (SOLUSI ERROR) ---
def preprocess_fit(df: pd.DataFrame, target_col: str = "risk_score"):
    """
    Helper function agar kompatibel dengan kode yang memanggil 'preprocess_fit'.
    Mengembalikan object pipeline agar bisa disimpan.
    """
    pp = PreprocessingPipeline()
    X, y = pp.fit(df, target_col)
    return pp, X, y # Return pipeline object juga, bukan hanya X dan y

def preprocess_transform(df: pd.DataFrame):
    """
    Helper function untuk memuat pipeline yang sudah disimpan dan mengubah data baru.
    Digunakan saat prediksi (prediction_service).
    """
    # Pastikan path ini sesuai dengan tempat kamu menyimpan artifacts
    base_dir = Path("ml/artifacts") 
    pipeline_path = base_dir / "preprocessing_pipeline.joblib"
    meta_path = base_dir / "preprocessing_meta.joblib"

    if not pipeline_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Artifacts preprocessing tidak ditemukan. Jalankan training/fit dulu.")

    pp = PreprocessingPipeline()
    pp.load(pipeline_path, meta_path)
    
    # Transform data
    X_transformed = pp.transform(df)
    
    # Kembalikan sebagai numpy array agar sesuai dengan input model.predict()
    return X_transformed.values

# CLI helper for quick testing
def run_preprocess_cli(csv_path: str, target_col: str = "risk_score", output_dir: str = "ml/artifacts") -> None:
    path = Path(csv_path)
    if not path.exists():
        logger.error("CSV not found: %s", csv_path)
        return

    df = pd.read_csv(path)
    pp = PreprocessingPipeline()
    X, y = pp.fit(df, target_col)
    saved = pp.save(output_dir)
    logger.info("Preprocessing finished. X shape: %s, y shape: %s", X.shape, y.shape)
    logger.info("Artifacts: %s", saved)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit preprocessing pipeline and save artifacts.")
    parser.add_argument("--csv", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", default="risk_score", help="Target column name")
    parser.add_argument("--out", default="ml/artifacts", help="Output directory for artifacts")
    args = parser.parse_args()

    run_preprocess_cli(args.csv, args.target, args.out)