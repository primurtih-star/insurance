from backend.core.risk_service import calculate_risk

def compute_risk(payload):
    age = payload["age"]
    bmi = payload["bmi"]
    smoker = payload["smoker"]

    score, level = calculate_risk(age, bmi, smoker)

    return {
        "score": score,
        "level": level
    }