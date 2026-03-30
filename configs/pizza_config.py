PIZZA_SURVEY_MAP = {
    "1": "https://concordia.yul1.qualtrics.com/jfe/form/SV_2cps8pBYmqBoJxk",
    "2": "https://concordia.yul1.qualtrics.com/jfe/form/SV_cCRzNs9C9udnXBs",
    "3": "https://concordia.yul1.qualtrics.com/jfe/form/SV_di1XtEdutYn62Ds",
}

PIZZA_MENTAL_MODEL_FEATURES = [
    "Maximum price",
    "Pizza style",
    "Ingredient preference",
    "Dietary restriction / allergy",
    "Importance of customer rating",
    "Importance of free delivery",
]

PIZZA_FEATURE_GROUP_MAP = {
    "max_price": "Maximum price",
    "pizza_style": "Pizza style",
    "ingredient_preference": "Ingredient preference",
    "dietary_restriction_model": "Dietary restriction / allergy",
    "rating_importance": "Importance of customer rating",
    "free_delivery_importance": "Importance of free delivery",
}

RATING_IMPORTANCE_TO_WEIGHT = {
    "Not important": 0.0,
    "Slightly important": 0.7,
    "Moderately important": 1.4,
    "Very important": 2.1,
    "Extremely important": 2.8,
}

FREE_DELIVERY_IMPORTANCE_TO_WEIGHT = {
    "Not important": 0.0,
    "Slightly important": 0.8,
    "Moderately important": 1.5,
    "Very important": 2.2,
    "Extremely important": 3.0,
}

DIETARY_OPTIONS = [
    "None",
    "Vegetarian",
    "Vegan",
    "Gluten-free",
    "Dairy-free",
    "Other (please specify)",
]


def normalize_dietary_for_model(user_choice: str) -> str:
    return "Other" if user_choice == "Other (please specify)" else user_choice


def result_formatter(payload: dict) -> str:
    meta = payload["meta"]
    dietary_note = ""
    if (
        payload["inputs"].get("dietary_restriction") == "Other (please specify)"
        and payload["inputs"].get("dietary_restriction_other_text", "").strip()
    ):
        dietary_note = f"\n- Reported dietary note: {payload['inputs']['dietary_restriction_other_text']}"

    return (
        f"**{meta['name']}**\n\n"
        f"- Style: {meta['style']}\n"
        f"- Main ingredient: {meta['ingredient']}\n"
        f"- Dietary tag: {meta['dietary_tag']}\n"
        f"- Customer rating: {meta['customer_rating']}\n"
        f"- Free delivery: {meta['free_delivery']}\n"
        f"- Price: ${meta['price']} CAD"
        f"{dietary_note}"
    )


def text_reason_builder(payload: dict) -> list[str]:
    meta = payload["meta"]
    inputs = payload["inputs"]
    reasons = []

    if meta["style"] == inputs["pizza_style"]:
        reasons.append(f"It matches your preferred pizza style: **{inputs['pizza_style']}**.")

    if meta["ingredient"] == inputs["ingredient_preference"]:
        reasons.append(f"It matches your preferred ingredient: **{inputs['ingredient_preference']}**.")

    if meta["price"] <= inputs["max_price"]:
        reasons.append(
            f"It stays within your budget limit of **${inputs['max_price']} CAD**."
        )
    else:
        reasons.append(
            f"It is close to your budget preference, even though it is above **${inputs['max_price']} CAD**."
        )

    if inputs["dietary_restriction"] != "None":
        reasons.append(
            f"It fits the dietary preference or restriction you selected: **{inputs['dietary_restriction']}**."
        )

    if inputs["rating_importance"] in ["Very important", "Extremely important"]:
        reasons.append(
            f"You said ratings matter a lot, and this pizza has a customer rating of **{meta['customer_rating']}**."
        )

    if inputs["free_delivery_importance"] in ["Very important", "Extremely important"] and meta["free_delivery"] == "Yes":
        reasons.append("You placed high importance on free delivery, and this option includes it.")

    if not reasons:
        reasons.append("This option best matched the combination of your style, ingredient, and budget preferences.")

    return reasons[:4]


PIZZA_CONFIG = {
    "task_name": "pizza",
    "bundle_path": "models/pizza_bundle.joblib",
    "survey_map": PIZZA_SURVEY_MAP,
    "mental_model_features": PIZZA_MENTAL_MODEL_FEATURES,
    "feature_group_map": PIZZA_FEATURE_GROUP_MAP,
    "result_title": "Recommended pizza",
    "max_shap_display": 10,
    "visual_caption": "This visual explanation shows the strongest factors that pushed the model toward this pizza recommendation.",
    "text_caption": "This text explanation summarizes the main reasons this pizza was recommended.",
    "result_formatter": result_formatter,
    "text_reason_builder": text_reason_builder,
}