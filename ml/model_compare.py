"""
Model Comparison Utilities.

Modul ini bertugas membandingkan performa dua model (Champion vs Candidate)
untuk menentukan apakah model baru layak dipromosikan ke produksi.

Fitur:
- Cold Start Handling (Jika belum ada Champion, Candidate otomatis menang).
- Metric Reuse (Menggunakan ml.utils.metrics).
- JSON Report Generation.
"""
from __future__ import annotations

import logging
import json
import joblib
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# --- INTEGRASI MODUL KITA ---
# Kita gunakan metrics yang sudah distandarisasi
from ml.utils.metrics import regression_metrics

logger = logging.getLogger(__name__)

def load_model_safe(path: Union[str, Path]) -> Any:
    """Helper untuk memuat model tanpa crash aplikasi."""
    path = Path(path)
    if not path.exists():
        logger.warning(f"⚠️ Model not found at: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        logger.error(f"❌ Failed to load model {path}: {e}")
        return None

def compare_models(
    champion_path: Path,
    candidate_path: Path,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    primary_metric: str = "r2_score"
) -> Tuple[Path, Dict[str, Any]]:
    """
    Membandingkan dua model berdasarkan Test Set.
    
    Args:
        champion_path: Path ke model produksi saat ini.
        candidate_path: Path ke model baru (hasil training).
        X_test, y_test: Data validasi.
        primary_metric: Metrik penentu kemenangan ('r2_score' atau 'rmse').
    
    Returns:
        (winner_path, report_dictionary)
    """
    logger.info(f"⚔️  DUEL START: Champion({champion_path.name}) vs Candidate({candidate_path.name})")

    # 1. Load Models
    model_champ = load_model_safe(champion_path)
    model_cand = load_model_safe(candidate_path)

    # Struktur Laporan Awal
    report: Dict[str, Any] = {
        "primary_metric": primary_metric,
        "champion": {"path": str(champion_path), "metrics": None},
        "candidate": {"path": str(candidate_path), "metrics": None},
        "winner": None,
        "improvement": 0.0,
        "reason": ""
    }

    # 2. Skenario: Champion Belum Ada (Cold Start)
    if model_champ is None:
        if model_cand is not None:
            logger.info("🎉 No champion exists. Candidate wins by default.")
            report["winner"] = str(candidate_path)
            report["reason"] = "champion_missing"
            # Hitung metrik candidate untuk laporan
            try:
                y_pred = model_cand.predict(X_test)
                report["candidate"]["metrics"] = regression_metrics(y_test, y_pred)
            except:
                pass
            return candidate_path, report
        else:
            logger.error("❌ Both models are missing/broken.")
            raise FileNotFoundError("No models available to compare.")

    # 3. Skenario: Candidate Gagal Load
    if model_cand is None:
        logger.info("🛡️ Candidate missing/broken. Champion remains.")
        report["winner"] = str(champion_path)
        report["reason"] = "candidate_missing"
        return champion_path, report

    # 4. Evaluasi (Hitung Skor)
    try:
        # Prediksi Champion
        y_pred_champ = model_champ.predict(X_test)
        metrics_champ = regression_metrics(y_test, y_pred_champ)
        report["champion"]["metrics"] = metrics_champ

        # Prediksi Candidate
        y_pred_cand = model_cand.predict(X_test)
        metrics_cand = regression_metrics(y_test, y_pred_cand)
        report["candidate"]["metrics"] = metrics_cand

        # 5. Tentukan Pemenang
        # Kita ambil nilai metrik, default ke angka buruk jika gagal hitung
        score_champ = metrics_champ.get(primary_metric, -999.0)
        score_cand = metrics_cand.get(primary_metric, -999.0)
        
        # Untuk RMSE/MAE, "lebih kecil lebih baik", untuk R2 "lebih besar lebih baik"
        # Di sini asumsi kita pakai R2 Score (Makin besar makin bagus)
        
        logger.info(f"   Scores ({primary_metric}): Champion={score_champ:.4f} | Candidate={score_cand:.4f}")

        if score_cand > score_champ:
            # Candidate Menang
            improvement = score_cand - score_champ
            logger.info(f"🚀 Candidate WINS! Improvement: +{improvement:.4f}")
            
            report["winner"] = str(candidate_path)
            report["improvement"] = round(improvement, 6)
            report["reason"] = "better_score"
            return candidate_path, report
        else:
            # Champion Bertahan
            logger.info("🛡️ Champion DEFENDS title. Candidate discarded.")
            
            report["winner"] = str(champion_path)
            report["improvement"] = round(score_cand - score_champ, 6)
            report["reason"] = "worse_score"
            return champion_path, report

    except Exception as e:
        logger.exception(f"Comparison failed due to runtime error: {e}")
        # Default ke Champion jika terjadi crash saat prediksi
        return champion_path, {"error": str(e)}