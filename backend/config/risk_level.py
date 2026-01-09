def classify_risk(score):
    if score >= 0.8:
        return "High Risk"
    elif score >= 0.5:
        return "Moderate Risk"
    return "Low Risk"