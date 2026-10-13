REQUIRED_COLUMNS = [
    "case_id",
    "full_name_raw",
    "full_name_normalized",
    "name_tokens",
    "disappearance_city_verified",
    "latitude",
    "longitude",
    "disappearance_date_parsed",
]

RECOMMENDED_COLUMNS = [
    "first_name_raw",
    "first_name_normalized",
    "last_name_raw",
    "last_name_normalized",
    "disappearance_city_raw",
    "disappearance_city_normalized",
    "disappearance_department_verified",
    "disappearance_date_raw",
    "city_resolution_status",
    "city_resolution_score",
    "city_resolution_explanation",
]

INVESTIGATION_MODE_SINGLE = "single_case"
INVESTIGATION_MODE_GLOBAL = "global"

