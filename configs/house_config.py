HOUSE_SURVEY_MAP = {
    "1": "https://concordia.yul1.qualtrics.com/jfe/form/SV_2cps8pBYmqBoJxk",
    "2": "https://concordia.yul1.qualtrics.com/jfe/form/SV_0BLYh0VPj7VxGt0",
    "3": "https://concordia.yul1.qualtrics.com/jfe/form/SV_4TJpNpKcMW1PpB4",
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
    top = payload["xai_agg"].head(6)["study_feature"].tolist()

    templates = {
        "Budget": f"Your budget (**${inputs['budget']} CAD**) was a key factor in narrowing the property options.",
        "City": f"Your preferred city (**{inputs['city']}**) strongly influenced the recommendation.",
        "Property type": f"Your preferred property type (**{inputs['property_type']}**) influenced the selected listing.",
        "Bedrooms": f"Your bedroom preference (**{inputs['bedrooms']}**) contributed to the recommendation.",
        "Bathrooms": f"Your bathroom preference (**{inputs['bathrooms']}**) contributed to the recommendation.",
        "Area size": f"Your minimum preferred area size (**{inputs['area_size']} sq ft**) influenced the final match.",
        "Distance to downtown": f"Your preferred distance to downtown (**{inputs['distance_to_downtown']}**) was taken into account.",
        "Public transport access": f"Your preference for public transport access (**{inputs['public_transport_access']}**) influenced the recommendation.",
        "School quality": f"Your school-quality preference (**{inputs['school_quality']}**) contributed to the final choice.",
        "Safety": f"Your stated importance of safety (**{inputs['safety']}**) affected the recommendation.",
        "Noise level": f"Your noise tolerance (**{inputs['noise_level']}**) influenced the decision.",
        "Parking": f"Your parking preference (**{inputs['parking']}**) was considered by the model.",
        "Garden": f"Your garden or yard preference (**{inputs['garden']}**) influenced the selected property.",
        "View quality": f"Your preferred view quality (**{inputs['view_quality']}**) contributed to the final recommendation.",
        "Building age": f"Your building age preference (**{inputs['building_age']}**) was taken into account.",
        "Investment potential": f"Your interest in investment potential (**{inputs['investment_potential']}**) influenced the result.",
        "Property tax sensitivity": f"Your property tax sensitivity (**{inputs['property_tax_sensitivity']}**) was considered in the decision.",
        "Family suitability": f"Your family suitability preference (**{inputs['family_suitability']}**) contributed to the recommendation.",
    }

    reasons = [templates[f] for f in top if f in templates]

    if not reasons:
        reasons.append("This property was the strongest overall match for your stated housing preferences.")

    return reasons[:6]


HOUSE_CONFIG = {
    "task_name": "house",
    "bundle_path": "models/house_bundle.joblib",
    "survey_map": HOUSE_SURVEY_MAP,
    "mental_model_features": HOUSE_MENTAL_MODEL_FEATURES,
    "feature_group_map": HOUSE_FEATURE_GROUP_MAP,
    "result_title": "Recommended house",
    "min_shap_display": 4,
    "max_shap_display": 6,
    "max_text_reasons": 6,
    "visual_caption": "This explanation summarizes the main preference factors the model used when selecting the recommended property.",
    "text_caption": "This explanation summarizes the main preference factors that influenced the property recommendation.",
    "result_formatter": result_formatter,
    "text_reason_builder": text_reason_builder,
}