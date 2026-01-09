import time
import os
import math
import pandas as pd
import logging
import traceback
from flask import Blueprint, jsonify, render_template

# Scikit-Learn Ecosystem
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Algorithms (Armory)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ==========================================
# 1. KONFIGURASI & BLUEPRINT
# ==========================================
bp = Blueprint("compare_model", __name__)
DATASET_PATH = os.path.join("data", "raw", "uploaded_dataset.csv")

# Setup Logger khusus Arena
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("⚔️ ModelArena")

# ==========================================
# 2. CLASS: THE GLADIATOR ARENA
# ==========================================
class ModelArena:
    """
    Engine canggih untuk mengadu performa berbagai algoritma ML.
    Fitur: Auto-Preprocessing, Fail-Safe Execution, & Time Benchmarking.
    """
    def __init__(self, df):
        self.df = df
        self.results = []
        
        # Definisi Fitur (Explicit Schema)
        self.numeric_features = ["age", "bmi", "children"]
        self.categorical_features = ["sex", "smoker", "region"]
        self.target = "charges"

    def _get_preprocessor(self):
        """Membangun Pipeline Preprocessing Standar Industri."""
        # 1. Numerik: Scaling (Wajib untuk SVR/Linear/KNN)
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        # 2. Kategorikal: One-Hot Encoding (Agar mesin mengerti teks)
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # 3. Gabungkan
        return ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

    def fight(self):
        """Memulai pertarungan antar model."""
        
        # 1. Validasi Integritas Data
        required_cols = self.numeric_features + self.categorical_features + [self.target]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Dataset tidak valid! Kolom hilang: {missing}")

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # 2. Split Data (80% Latihan, 20% Ujian)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Persiapkan Senjata (Daftar Model)
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Lasso Regression": Lasso(alpha=0.1),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "SVR (Support Vector)": SVR(kernel='rbf', C=100, gamma=0.1), # Tuned params
            "KNN Regressor": KNeighborsRegressor(n_neighbors=5)
        }

        preprocessor = self._get_preprocessor()
        arena_results = []

        logger.info(f"⚡ Memulai pertarungan dengan {len(models)} model...")

        # 4. Eksekusi Pertarungan
        for name, model in models.items():
            try:
                # Bungkus dalam Pipeline (Raw Data -> Clean Data -> Model)
                clf = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', model)])
                
                # --- START TIMER ---
                start_time = time.time()
                
                clf.fit(X_train, y_train)   # Training
                preds = clf.predict(X_test) # Testing
                
                # --- STOP TIMER ---
                elapsed_time = time.time() - start_time

                # Hitung Skor
                mae = mean_absolute_error(y_test, preds)
                rmse = math.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)

                # Simpan Hasil
                arena_results.append({
                    "model": name,
                    "mae": round(mae, 2),
                    "rmse": round(rmse, 2),
                    "r2": round(r2, 4),         # 4 desimal untuk presisi
                    "time_sec": round(elapsed_time, 4) # Kecepatan eksekusi
                })
                
                logger.info(f"✅ {name}: R2={r2:.4f} | Time={elapsed_time:.4f}s")

            except Exception as e:
                logger.error(f"❌ {name} CRASHED: {str(e)}")
                # Jangan biarkan error satu model menghentikan semuanya
                arena_results.append({
                    "model": name,
                    "r2": -1.0, # Penalti error
                    "error": "Model failed to converge"
                })

        # 5. Tentukan Pemenang (Sort by R2 Highest)
        self.results = sorted(arena_results, key=lambda k: k.get('r2', -1), reverse=True)
        return self.results


# ==========================================
# 3. ROUTE DEFINITIONS
# ==========================================

# --- A. API ENDPOINT (Untuk Data JSON) ---
# Frontend JavaScript akan fetch ke sini
@bp.route("/api/compare", methods=["GET"])
def compare_models_api():
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "Dataset belum diupload! Silakan ke menu Dataset Manager."}), 404

    try:
        df = pd.read_csv(DATASET_PATH)
        
        # Masuk ke Arena
        arena = ModelArena(df)
        results = arena.fight()
        
        # Analisis Juara
        champion = results[0] if results else None
        
        response = {
            "status": "success",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_rows": len(df),
            "champion": champion['model'] if champion else "None",
            "champion_r2": champion['r2'] if champion else 0,
            "leaderboard": results
        }

        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Critical System Failure: {traceback.format_exc()}")
        return jsonify({"error": "Internal Server Error saat training model."}), 500


# --- B. VIEW ENDPOINT (Untuk Tampilan HTML) ---
# Sidebar akan mengarah ke sini
@bp.route("/model-arena")
def view_model_arena():
    # Render file HTML yang sudah kamu buat sebelumnya
    return render_template("compare.html", title="Model Battle Arena")