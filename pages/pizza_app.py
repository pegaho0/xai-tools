import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st

from app_core import (
    aggregate_shap_to_study_features,
    build_return_url,
    compute_shap_for_row,
    hide_sidebar_nav,
    init_result_state,
    parse_cad_input,
    render_cad_text_input,
    render_choice_field,
    render_generic_result,
    render_mental_model_rating,
    timestamp_now,
    validate_and_store_route,
)
from configs.pizza_config import (
    DIETARY_OPTIONS,
    PIZZA_CONFIG,
    normalize_dietary_for_model,
)

st.set_page_config(page_title="Pizza Recommendation Study", layout="centered")
hide_sidebar_nav()
route = validate_and_store_route()

from model_loader import load_model_bundle
bundle = load_model_bundle("pizza")

catalog = pd.read_csv("data/pizza_catalog.csv")
pizza_id_to_meta = {row["pizza_id"]: row.to_dict() for _, row in catalog.iterrows()}

result_ready_key, result_payload_key, mm_rating_key = init_result_state("pizza")

st.title("🍕 Pizza Recommendation")
st.caption(f"{'Visual Explanation' if route['group'] == 'visual' else 'Text Explanation'} • Step {route['step']} of 3")
st.caption(
    "Enter your preferences, rate how important you think each factor is to the AI, "
    "get a recommendation, and continue to the survey."
)

max_price_text, max_price = render_cad_text_input(
    "What is the maximum price you are willing to pay (CAD)? *",
    key="pizza_max_price_text",
    placeholder="e.g. 25 or CAD 25",
)

pizza_style = render_choice_field(
    "Which pizza style do you prefer? *",
    ["Italian", "American"],
    key="pizza_style",
)

ingredient_preference = render_choice_field(
    "Which ingredient do you prefer most? *",
    ["Cheese", "Pepperoni", "Chicken", "Vegetables", "Mushrooms"],
    key="ingredient_preference",
)

dietary_restriction = render_choice_field(
    "Do you have any dietary restriction or allergy? *",
    DIETARY_OPTIONS,
    key="dietary_restriction",
    horizontal=False,
)

dietary_restriction_other_text = ""
if dietary_restriction == "Other (please specify)":
    dietary_restriction_other_text = st.text_input("Please specify your dietary restriction or allergy *")

rating_importance = render_choice_field(
    "How important are customer ratings when choosing a pizza? *",
    ["Not important", "Slightly important", "Moderately important", "Very important", "Extremely important"],
    key="rating_importance",
    horizontal=False,
)

free_delivery_importance = render_choice_field(
    "How important is free delivery when choosing a pizza? *",
    ["Not important", "Slightly important", "Moderately important", "Very important", "Extremely important"],
    key="free_delivery_importance",
    horizontal=False,
)

mental_model_ratings, all_answered = render_mental_model_rating(
    PIZZA_CONFIG["mental_model_features"],
    mm_rating_key,
)

submit = st.button("Get recommendation", type="primary", use_container_width=True)

if submit:
    input_errors = []

    if max_price is None:
        if str(max_price_text).strip() == "":
            input_errors.append("Maximum price is required.")
        else:
            input_errors.append("Maximum price must be a valid amount in CAD.")
    if pizza_style is None:
        input_errors.append("Pizza style is required.")
    if ingredient_preference is None:
        input_errors.append("Ingredient preference is required.")
    if dietary_restriction is None:
        input_errors.append("Dietary restriction or allergy is required.")
    if dietary_restriction == "Other (please specify)" and not dietary_restriction_other_text.strip():
        input_errors.append("Please specify your dietary restriction or allergy.")
    if rating_importance is None:
        input_errors.append("Customer rating importance is required.")
    if free_delivery_importance is None:
        input_errors.append("Free delivery importance is required.")
    if not all_answered:
        input_errors.append("Please rate all factors before continuing.")

    if input_errors:
        for err in input_errors:
            st.error(err)
    else:
        dietary_restriction_model = normalize_dietary_for_model(dietary_restriction)

        x = pd.DataFrame([{
            "max_price": max_price,
            "pizza_style": pizza_style,
            "ingredient_preference": ingredient_preference,
            "dietary_restriction_model": dietary_restriction_model,
            "rating_importance": rating_importance,
            "free_delivery_importance": free_delivery_importance,
        }])

        pred_id = bundle["model"].predict(x)[0]
        meta = pizza_id_to_meta[pred_id]

        _, base_value, shap_df = compute_shap_for_row(bundle, x)
        xai_agg = aggregate_shap_to_study_features(shap_df, PIZZA_CONFIG["feature_group_map"])
        xai_rank_list = xai_agg["study_feature"].tolist()

        st.session_state[result_ready_key] = True
        st.session_state[result_payload_key] = {
            "timestamp": timestamp_now(),
            "inputs": {
                "max_price": max_price,
                "pizza_style": pizza_style,
                "ingredient_preference": ingredient_preference,
                "dietary_restriction": dietary_restriction,
                "dietary_restriction_other_text": dietary_restriction_other_text,
                "rating_importance": rating_importance,
                "free_delivery_importance": free_delivery_importance,
            },
            "mental_model_ratings": mental_model_ratings.copy(),
            "recommended_id": pred_id,
            "recommended_name": meta["name"],
            "meta": meta,
            "shap_df": shap_df,
            "base_value": base_value,
            "xai_agg": xai_agg,
            "xai_rank_list": xai_rank_list,
        }
        st.rerun()

if st.session_state[result_ready_key] and st.session_state[result_payload_key] is not None:
    payload = st.session_state[result_payload_key]
    render_generic_result(route, PIZZA_CONFIG, payload)
    return_url = build_return_url(route, PIZZA_CONFIG["survey_map"], payload, PIZZA_CONFIG["task_name"])
    st.link_button("Continue to Survey", return_url, use_container_width=True)
else:
    st.caption("Complete all sections and click Get recommendation to continue.")