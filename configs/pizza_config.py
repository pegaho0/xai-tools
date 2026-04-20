PIZZA_SURVEY_MAP = {
    "1": "https://concordia.yul1.qualtrics.com/jfe/form/SV_2cps8pBYmqBoJxk",
    "2": "https://concordia.yul1.qualtrics.com/jfe/form/SV_0BLYh0VPj7VxGt0",
    "3": "https://concordia.yul1.qualtrics.com/jfe/form/SV_4TJpNpKcMW1PpB4",
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
    top = payload["xai_agg"].head(6)["study_feature"].tolist()

    templates = {
        "Maximum price": (
            f"Your budget limit (**${inputs['max_price']} CAD**) influenced which pizzas were strong matches."
        ),
        "Pizza style": (
            f"Your preferred pizza style (**{inputs['pizza_style']}**) contributed to this recommendation."
        ),
        "Ingredient preference": (
            f"Your preferred ingredient (**{inputs['ingredient_preference']}**) was an important factor in the model’s decision."
        ),
        "Dietary restriction / allergy": (
            f"Your dietary selection (**{inputs['dietary_restriction']}**) affected which options were suitable."
        ),
        "Importance of customer rating": (
            f"Your stated importance of ratings (**{inputs['rating_importance']}**) influenced how the model weighed customer scores."
        ),
        "Importance of free delivery": (
            f"Your stated importance of free delivery (**{inputs['free_delivery_importance']}**) contributed to the final recommendation."
        ),
    }

    reasons = [templates[f] for f in top if f in templates]

    if not reasons:
        reasons.append("This pizza was the strongest overall match for the combination of your stated preferences.")

    return reasons[:6]


PIZZA_CONFIG = {
    "task_name": "pizza",
    "bundle_path": "models/pizza_bundle.joblib",
    "survey_map": PIZZA_SURVEY_MAP,
    "mental_model_features": PIZZA_MENTAL_MODEL_FEATURES,
    "feature_group_map": PIZZA_FEATURE_GROUP_MAP,
    "result_title": "Recommended pizza",
    "min_shap_display": 4,
    "max_shap_display": 6,
    "max_text_reasons": 6,
    "visual_caption": "This explanation summarizes the main preference factors the model used when selecting the recommended pizza.",
    "text_caption": "This explanation summarizes the main preference factors that influenced the pizza recommendation.",
    "result_formatter": result_formatter,
    "text_reason_builder": text_reason_builder,
}