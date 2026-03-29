import pandas as pd
import streamlit as st

from app_core import (
    aggregate_shap_to_study_features,
    build_return_url,
    compute_shap_for_row,
    hide_sidebar_nav,
    init_result_state,
    load_bundle,
    ranking_list_to_rank_dict,
    render_generic_result,
    render_mental_model_sort,
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

bundle = load_bundle(PIZZA_CONFIG["bundle_path"])
catalog = pd.read_csv("data/pizza_catalog.csv")
pizza_id_to_meta = {row["pizza_id"]: row.to_dict() for _, row in catalog.iterrows()}

result_ready_key, result_payload_key, mm_order_key = init_result_state("pizza")

st.title("🍕 Pizza Recommendation")
st.caption(f"{'Visual Explanation' if route['group'] == 'visual' else 'Text Explanation'} • Step {route['step']} of 3")
st.caption(
    "Enter your preferences, rank the factors you think matter most to the AI, "
    "get a recommendation, and continue to the survey."
)

max_price = st.slider("What is the maximum price you are willing to pay (CAD)?", 15, 50, 25, 1)
pizza_style = st.radio("Which pizza style do you prefer?", ["Italian", "American"], horizontal=True)
ingredient_preference = st.selectbox(
    "Which ingredient do you prefer most?",
    ["Cheese", "Pepperoni", "Chicken", "Vegetables", "Mushrooms"],
    index=0,
)
dietary_restriction = st.selectbox(
    "Do you have any dietary restriction or allergy?",
    DIETARY_OPTIONS,
    index=0,
)

dietary_restriction_other_text = ""
if dietary_restriction == "Other (please specify)":
    dietary_restriction_other_text = st.text_input("Please specify your dietary restriction or allergy")

rating_importance = st.selectbox(
    "How important are customer ratings when choosing a pizza?",
    ["Not important", "Slightly important", "Moderately important", "Very important", "Extremely important"],
    index=2,
)
free_delivery_importance = st.selectbox(
    "How important is free delivery when choosing a pizza?",
    ["Not important", "Slightly important", "Moderately important", "Very important", "Extremely important"],
    index=2,
)

dietary_restriction_model = normalize_dietary_for_model(dietary_restriction)

x = pd.DataFrame([{
    "max_price": max_price,
    "pizza_style": pizza_style,
    "ingredient_preference": ingredient_preference,
    "dietary_restriction_model": dietary_restriction_model,
    "rating_importance": rating_importance,
    "free_delivery_importance": free_delivery_importance,
}])

mental_model_order = render_mental_model_sort(PIZZA_CONFIG["mental_model_features"], mm_order_key)
mental_model_ranks = ranking_list_to_rank_dict(mental_model_order)

if st.button("Get recommendation", type="primary", use_container_width=True):
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
        "mental_model_order": mental_model_order.copy(),
        "mental_model_ranks": mental_model_ranks,
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