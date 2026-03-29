HOUSE_SURVEY_MAP = {
    "1": "https://concordia.yul1.qualtrics.com/jfe/form/SV_2cps8pBYmqBoJxk",
    "2": "https://concordia.yul1.qualtrics.com/jfe/form/SV_cCRzNs9C9udnXBs",
    "3": "https://concordia.yul1.qualtrics.com/jfe/form/SV_di1XtEdutYn62Ds",
}

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

HOUSE_CONFIG = {
    "task_name": "house",
    "bundle_path": "models/house_bundle.joblib",
    "survey_map": HOUSE_SURVEY_MAP,
    "mental_model_features": HOUSE_MENTAL_MODEL_FEATURES,
    "feature_group_map": HOUSE_FEATURE_GROUP_MAP,
    "result_title": "Recommended house",
    "max_shap_display": 12,
    "visual_caption": "This visual explanation shows which inputs had the strongest influence on the house recommendation.",
    "text_caption": "This text explanation summarizes the most influential factors behind the house recommendation.",
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


HOUSE_CONFIG["result_formatter"] = result_formatter