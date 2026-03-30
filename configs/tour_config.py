TOUR_SURVEY_MAP = {
    "1": "https://concordia.yul1.qualtrics.com/jfe/form/SV_2cps8pBYmqBoJxk",
    "2": "https://concordia.yul1.qualtrics.com/jfe/form/SV_cCRzNs9C9udnXBs",
    "3": "https://concordia.yul1.qualtrics.com/jfe/form/SV_di1XtEdutYn62Ds",
}

TOUR_MENTAL_MODEL_FEATURES = [
    "Budget",
    "Trip duration",
    "Preferred region",
    "Preferred climate",
    "Travel style",
    "Group type",
    "Accommodation level",
    "Food interest",
    "Transportation comfort",
    "Season",
    "Safety importance",
    "Rating importance",
]

TOUR_FEATURE_GROUP_MAP = {
    "budget": "Budget",
    "trip_duration": "Trip duration",
    "preferred_region": "Preferred region",
    "preferred_climate": "Preferred climate",
    "travel_style": "Travel style",
    "group_type": "Group type",
    "accommodation_level": "Accommodation level",
    "food_interest": "Food interest",
    "transportation_comfort": "Transportation comfort",
    "season": "Season",
    "safety_importance": "Safety importance",
    "rating_importance": "Rating importance",
}


def result_formatter(payload: dict) -> str:
    meta = payload["meta"]
    return (
        f"**{meta['tour_name']}**\n\n"
        f"- Region: {meta['region']}\n"
        f"- Climate: {meta['climate']}\n"
        f"- Travel style: {meta['travel_style']}\n"
        f"- Group fit: {meta['group_type']}\n"
        f"- Duration: {meta['trip_duration']}\n"
        f"- Price: ${meta['price']} CAD\n"
        f"- Rating: {meta['rating']}"
    )


def text_reason_builder(payload: dict) -> list[str]:
    meta = payload["meta"]
    inputs = payload["inputs"]
    reasons = []

    if meta["region"] == inputs["preferred_region"]:
        reasons.append(f"It matches your preferred region: **{inputs['preferred_region']}**.")

    if meta["climate"] == inputs["preferred_climate"]:
        reasons.append(f"It matches your preferred climate: **{inputs['preferred_climate']}**.")

    if meta["travel_style"] == inputs["travel_style"]:
        reasons.append(f"It fits your preferred travel style: **{inputs['travel_style']}**.")

    if meta["trip_duration"] == inputs["trip_duration"]:
        reasons.append(f"It matches your preferred trip duration: **{inputs['trip_duration']}**.")

    if meta["price"] <= inputs["budget"]:
        reasons.append(f"It fits within your budget of **${inputs['budget']} CAD**.")
    else:
        reasons.append(f"It is one of the closest matches to your budget of **${inputs['budget']} CAD**.")

    if inputs["rating_importance"] in ["High", "Very high"]:
        reasons.append(f"You gave importance to ratings, and this tour has a rating of **{meta['rating']}**.")

    if not reasons:
        reasons.append("This tour was the strongest overall match for your travel preferences.")

    return reasons[:4]


TOUR_CONFIG = {
    "task_name": "tour",
    "bundle_path": "models/tour_bundle.joblib",
    "survey_map": TOUR_SURVEY_MAP,
    "mental_model_features": TOUR_MENTAL_MODEL_FEATURES,
    "feature_group_map": TOUR_FEATURE_GROUP_MAP,
    "result_title": "Recommended tour",
    "max_shap_display": 12,
    "visual_caption": "This visual explanation shows the strongest factors that supported this tour recommendation.",
    "text_caption": "This text explanation summarizes the main reasons this tour was recommended.",
    "result_formatter": result_formatter,
    "text_reason_builder": text_reason_builder,
}