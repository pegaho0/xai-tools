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
    meta = payload["meta"]
    inputs = payload["inputs"]
    reasons = []

    if meta["city"] == inputs["city"]:
        reasons.append(f"It matches your preferred city: **{inputs['city']}**.")

    if meta["property_type"] == inputs["property_type"]:
        reasons.append(f"It matches your preferred property type: **{inputs['property_type']}**.")

    if meta["price"] <= inputs["budget"]:
        reasons.append(f"It fits within your budget of **${inputs['budget']} CAD**.")
    else:
        reasons.append(f"It is one of the closest matches to your budget of **${inputs['budget']} CAD**.")

    if meta["bedrooms"] >= inputs["bedrooms"]:
        reasons.append(f"It meets your bedroom preference with **{meta['bedrooms']} bedrooms**.")

    if meta["bathrooms"] >= inputs["bathrooms"]:
        reasons.append(f"It meets your bathroom preference with **{meta['bathrooms']} bathrooms**.")

    if inputs["parking"] == "Yes" and meta["parking"] == "Yes":
        reasons.append("You asked for parking, and this property includes parking.")

    if inputs["garden"] == "Yes" and meta["garden"] == "Yes":
        reasons.append("You asked for a garden or yard, and this property includes one.")

    if not reasons:
        reasons.append("This property was the strongest overall match for your housing preferences.")

    return reasons[:4]


HOUSE_CONFIG = {
    "task_name": "house",
    "bundle_path": "models/house_bundle.joblib",
    "survey_map": HOUSE_SURVEY_MAP,
    "mental_model_features": HOUSE_MENTAL_MODEL_FEATURES,
    "feature_group_map": HOUSE_FEATURE_GROUP_MAP,
    "result_title": "Recommended house",
    "max_shap_display": 12,
    "visual_caption": "This visual explanation shows the strongest factors that pushed the model toward this house recommendation.",
    "text_caption": "This text explanation summarizes the main reasons this house was recommended.",
    "result_formatter": result_formatter,
    "text_reason_builder": text_reason_builder,
}