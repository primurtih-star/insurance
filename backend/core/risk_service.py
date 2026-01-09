import numpy as np


# =============================================================
# 1. FUNGSI UTAMA: HITUNG RISK SCORE
# =============================================================

def calculate_risk_score(data: dict):
    """
    Input:
    {
        "age": 29,
        "bmi": 27.3,
        "children": 1,
        "smoker": "yes",
        "region": "southeast",
        "charges": 16884.0
    }
    """

    age = float(data.get("age", 0))
    bmi = float(data.get("bmi", 0))
    children = float(data.get("children", 0))
    smoker = data.get("smoker", "no")
    charges = float(data.get("charges", 0))

    # ---------------------------------------------------------
    # RISK COMPONENTS (dibuat actuarial style)
    # ---------------------------------------------------------

    # Age Risk (skala 0–30)
    age_risk = min(age / 3.5, 30)

    # BMI Risk (skala 0–25)
    bmi_risk = min(max((bmi - 18.5) * 1.5, 0), 25)

    # Smoking Risk (skala 0–30)
    smoker_risk = 30 if smoker.lower() == "yes" else 5

    # Children Risk (skala 0–10)
    children_risk = min(children * 2, 10)

    # Medical Cost Risk (skala 0–20)
    cost_risk = min(charges / 2000, 20)

    # ---------------------------------------------------------
    # TOTAL SCORE
    # ---------------------------------------------------------

    total_risk = age_risk + bmi_risk + smoker_risk + children_risk + cost_risk
    total_risk = min(total_risk, 100)     # cap di 100

    return {
        "score": round(total_risk, 2),
        "category": classify_risk(total_risk),
        "components": {
            "age_risk": round(age_risk, 2),
            "bmi_risk": round(bmi_risk, 2),
            "smoker_risk": round(smoker_risk, 2),
            "children_risk": round(children_risk, 2),
            "cost_risk": round(cost_risk, 2),
        }
    }


# =============================================================
# 2. KATEGORI RISIKO
# =============================================================

def classify_risk(score: float):
    if score < 35:
        return "LOW"
    elif 35 <= score < 70:
        return "MEDIUM"
    else:
        return "HIGH"
