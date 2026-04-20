TOUR_SURVEY_MAP = {
    "1": "https://concordia.yul1.qualtrics.com/jfe/form/SV_2cps8pBYmqBoJxk",
    "2": "https://concordia.yul1.qualtrics.com/jfe/form/SV_0BLYh0VPj7VxGt0",
    "3": "https://concordia.yul1.qualtrics.com/jfe/form/SV_4TJpNpKcMW1PpB4",
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
    inputs = payload["inputs"]
    top = payload["xai_agg"].head(6)["study_feature"].tolist()

    templates = {
        "Budget": f"Your budget (**${inputs['budget']} CAD**) shaped which tours were realistic matches.",
        "Trip duration": f"Your preferred trip duration (**{inputs['trip_duration']}**) influenced the recommendation.",
        "Preferred region": f"Your preferred region (**{inputs['preferred_region']}**) was an important factor in the selection.",
        "Preferred climate": f"Your preferred climate (**{inputs['preferred_climate']}**) influenced the final recommendation.",
        "Travel style": f"Your travel style (**{inputs['travel_style']}**) strongly affected the selected tour.",
        "Group type": f"Who you are travelling with (**{inputs['group_type']}**) contributed to the model’s decision.",
        "Accommodation level": f"Your accommodation preference (**{inputs['accommodation_level']}**) was taken into account.",
        "Food interest": f"Your interest in local food experiences (**{inputs['food_interest']}**) influenced the recommendation.",
        "Transportation comfort": f"Your transportation comfort preference (**{inputs['transportation_comfort']}**) contributed to the decision.",
        "Season": f"Your preferred season (**{inputs['season']}**) influenced the model’s choice.",
        "Safety importance": f"Your stated importance of safety (**{inputs['safety_importance']}**) affected the recommendation.",
        "Rating importance": f"Your stated importance of ratings (**{inputs['rating_importance']}**) influenced how the model weighed tour quality.",
    }

    reasons = [templates[f] for f in top if f in templates]

    if not reasons:
        reasons.append("This tour was the strongest overall match for your stated travel preferences.")

    return reasons[:6]


TOUR_CONFIG = {
    "task_name": "tour",
    "bundle_path": "models/tour_bundle.joblib",
    "survey_map": TOUR_SURVEY_MAP,
    "mental_model_features": TOUR_MENTAL_MODEL_FEATURES,
    "feature_group_map": TOUR_FEATURE_GROUP_MAP,
    "result_title": "Recommended tour",
    "min_shap_display": 4,
    "max_shap_display": 6,
    "max_text_reasons": 6,
    "visual_caption": "This explanation summarizes the main preference factors the model used when selecting the recommended tour.",
    "text_caption": "This explanation summarizes the main preference factors that influenced the tour recommendation.",
    "result_formatter": result_formatter,
    "text_reason_builder": text_reason_builder,
}