HOUSE_SURVEY_MAP = {
    "1": "https://concordia.yul1.qualtrics.com/jfe/form/SV_2cps8pBYmqBoJxk",
    "2": "https://concordia.yul1.qualtrics.com/jfe/form/SV_cCRzNs9C9udnXBs",
    "3": "https://concordia.yul1.qualtrics.com/jfe/form/SV_di1XtEdutYn62Ds",
}

HOUSE_MENTAL_MODEL_FEATURES = [
    "Budget",
    "City",
    "Property type",
    "Bedrooms",
    "Bathrooms",
    "Area size",
    "Distance to downtown",
    "Public transport access",
    "School quality",
    "Safety",
    "Noise level",
    "Parking",
    "Garden",
    "View quality",
    "Building age",
    "Investment potential",
    "Property tax sensitivity",
    "Family suitability",
]

HOUSE_FEATURE_GROUP_MAP = {
    "budget": "Budget",
    "city": "City",
    "property_type": "Property type",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "area_size": "Area size",
    "distance_to_downtown": "Distance to downtown",
    "public_transport_access": "Public transport access",
    "school_quality": "School quality",
    "safety": "Safety",
    "noise_level": "Noise level",
    "parking": "Parking",
    "garden": "Garden",
    "view_quality": "View quality",
    "building_age": "Building age",
    "investment_potential": "Investment potential",
    "property_tax_sensitivity": "Property tax sensitivity",
    "family_suitability": "Family suitability",
}


def result_formatter(payload: dict) -> str:
    meta = payload["meta"]
    return (
        f"**{meta['listing_name']}**\n\n"
        f"- City: {meta['city']}\n"
        f"- Type: {meta['property_type']}\n"
        f"- Bedrooms: {meta['bedrooms']}\n"
        f"- Bathrooms: {meta['bathrooms']}\n"
        f"- Area: {meta['area_size']} sq ft\n"
        f"- Price: ${meta['price']} CAD\n"
        f"- Parking: {meta['parking']}\n"
        f"- Garden: {meta['garden']}\n"
        f"- View: {meta['view_quality']}"
    )


def text_reason_builder(payload: dict) -> list[str]:
    inputs = payload["inputs"]
    ranked = payload["xai_agg"]["study_feature"].tolist()

    templates = {
        "Budget": f"Your budget (**${inputs['budget']} CAD**) was a key factor in narrowing the available properties.",
        "City": f"Your preferred city (**{inputs['city']}**) strongly influenced the recommendation.",
        "Property type": f"Your preferred property type (**{inputs['property_type']}**) contributed to the selected listing.",
        "Bedrooms": f"Your bedroom preference (**{inputs['bedrooms']}**) was considered in the final match.",
        "Bathrooms": f"Your bathroom preference (**{inputs['bathrooms']}**) contributed to the recommendation.",
        "Area size": f"Your minimum preferred area size (**{inputs['area_size']} sq ft**) influenced the final choice.",
        "Distance to downtown": f"Your preferred distance to downtown (**{inputs['distance_to_downtown']}**) was considered by the model.",
        "Public transport access": f"Your public transport preference (**{inputs['public_transport_access']}**) contributed to the recommendation.",
        "School quality": f"Your school-quality preference (**{inputs['school_quality']}**) influenced the final selection.",
        "Safety": f"Your stated importance of safety (**{inputs['safety']}**) was considered in the decision.",
        "Noise level": f"Your noise tolerance (**{inputs['noise_level']}**) contributed to the fit of the property.",
        "Parking": f"Your parking preference (**{inputs['parking']}**) was one of the factors considered.",
        "Garden": f"Your garden or yard preference (**{inputs['garden']}**) influenced the recommendation.",
        "View quality": f"Your preferred view quality (**{inputs['view_quality']}**) contributed to the selected property.",
        "Building age": f"Your building age preference (**{inputs['building_age']}**) was taken into account.",
        "Investment potential": f"Your interest in investment potential (**{inputs['investment_potential']}**) influenced the recommendation.",
        "Property tax sensitivity": f"Your property tax sensitivity (**{inputs['property_tax_sensitivity']}**) was considered in the final choice.",
        "Family suitability": f"Your family suitability preference (**{inputs['family_suitability']}**) contributed to the match.",
    }

    reasons = [templates[f] for f in ranked if f in templates]
    if not reasons:
        reasons.append("This property was the strongest overall match for your housing preferences.")
    return reasons[:6]


HOUSE_CONFIG = {
    "task_name": "house",
    "bundle_path": "models/house_bundle.joblib",
    "survey_map": HOUSE_SURVEY_MAP,
    "mental_model_features": HOUSE_MENTAL_MODEL_FEATURES,
    "feature_group_map": HOUSE_FEATURE_GROUP_MAP,
    "result_title": "Recommended house",
    "min_features_to_show": 4,
    "max_shap_display": 6,
    "max_text_reasons": 6,
    "visual_caption": "This explanation summarizes the main housing preference signals the model used when selecting the recommended property.",
    "text_caption": "This explanation summarizes the main housing preference signals that influenced the property recommendation.",
    "result_formatter": result_formatter,
    "text_reason_builder": text_reason_builder,
}