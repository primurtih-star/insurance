APP_NAME = "Predicture"
VERSION = "1.0.0"
CURRENCY_SYMBOL = "$"

ML_CONSTRAINTS = {
    "age_min": 18,
    "age_max": 100,
    "bmi_min": 10.0,
    "bmi_max": 60.0,
    "children_max": 10,
}

GENDER_OPTIONS = ["male", "female"]
SMOKER_OPTIONS = ["yes", "no"]
REGION_OPTIONS = ["southwest", "southeast", "northwest", "northeast"]

CSV_REQUIRED_COLUMNS = [
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges",
]