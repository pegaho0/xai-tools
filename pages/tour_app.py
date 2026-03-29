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
from configs.tour_config import TOUR_CONFIG

st.set_page_config(page_title="Tour Recommendation Study", layout="centered")
hide_sidebar_nav()
route = validate_and_store_route()

bundle = load_bundle(TOUR_CONFIG["bundle_path"])
catalog = pd.read_csv("data/tour_catalog.csv")
tour_id_to_meta = {row["tour_id"]: row.to_dict() for _, row in catalog.iterrows()}

result_ready_key, result_payload_key, mm_order_key = init_result_state("tour")

st.title("🌍 Tour Recommendation")
st.caption(f"{'Visual Explanation' if route['group'] == 'visual' else 'Text Explanation'} • Step {route['step']} of 3")
st.caption(
    "Provide your travel preferences, rank the factors you think matter most to the AI, "
    "then get a tour recommendation."
)

budget = st.slider("What is your budget (CAD)?", 800, 6000, 2500, 100)
trip_duration = st.selectbox("Preferred trip duration", ["Short", "Medium", "Long"], index=1)
preferred_region = st.selectbox(
    "Preferred region",
    ["Europe", "Asia", "North America", "South America", "Middle East", "Africa"],
    index=0,
)
preferred_climate = st.selectbox("Preferred climate", ["Cold", "Mild", "Warm"], index=1)
travel_style = st.selectbox("Preferred travel style", ["Adventure", "Relaxation", "Culture", "Nature", "Mixed"], index=2)
group_type = st.selectbox("Who are you travelling with?", ["Solo", "Couple", "Family", "Friends"], index=1)
accommodation_level = st.selectbox("Preferred accommodation level", ["Budget", "Standard", "Premium", "Luxury"], index=1)
food_interest = st.selectbox("How important is local food experience?", ["Low", "Medium", "High"], index=1)
transportation_comfort = st.selectbox("Transportation comfort preference", ["Basic", "Moderate", "High"], index=1)
season = st.selectbox("Preferred season", ["Spring", "Summer", "Autumn", "Winter"], index=1)
safety_importance = st.selectbox("How important is safety?", ["Low", "Medium", "High", "Very high"], index=2)
rating_importance = st.selectbox("How important are tour ratings?", ["Low", "Medium", "High", "Very high"], index=2)

x = pd.DataFrame([{
    "budget": budget,
    "trip_duration": trip_duration,
    "preferred_region": preferred_region,
    "preferred_climate": preferred_climate,
    "travel_style": travel_style,
    "group_type": group_type,
    "accommodation_level": accommodation_level,
    "food_interest": food_interest,
    "transportation_comfort": transportation_comfort,
    "season": season,
    "safety_importance": safety_importance,
    "rating_importance": rating_importance,
}])

mental_model_order = render_mental_model_sort(TOUR_CONFIG["mental_model_features"], mm_order_key)
mental_model_ranks = ranking_list_to_rank_dict(mental_model_order)

if st.button("Get recommendation", type="primary", use_container_width=True):
    pred_id = bundle["model"].predict(x)[0]
    meta = tour_id_to_meta[pred_id]

    _, base_value, shap_df = compute_shap_for_row(bundle, x)
    xai_agg = aggregate_shap_to_study_features(shap_df, TOUR_CONFIG["feature_group_map"])
    xai_rank_list = xai_agg["study_feature"].tolist()

    st.session_state[result_ready_key] = True
    st.session_state[result_payload_key] = {
        "timestamp": timestamp_now(),
        "inputs": {
            "budget": budget,
            "trip_duration": trip_duration,
            "preferred_region": preferred_region,
            "preferred_climate": preferred_climate,
            "travel_style": travel_style,
            "group_type": group_type,
            "accommodation_level": accommodation_level,
            "food_interest": food_interest,
            "transportation_comfort": transportation_comfort,
            "season": season,
            "safety_importance": safety_importance,
            "rating_importance": rating_importance,
        },
        "mental_model_order": mental_model_order.copy(),
        "mental_model_ranks": mental_model_ranks,
        "recommended_id": pred_id,
        "recommended_name": meta["tour_name"],
        "meta": meta,
        "shap_df": shap_df,
        "base_value": base_value,
        "xai_agg": xai_agg,
        "xai_rank_list": xai_rank_list,
    }
    st.rerun()

if st.session_state[result_ready_key] and st.session_state[result_payload_key] is not None:
    payload = st.session_state[result_payload_key]
    render_generic_result(route, TOUR_CONFIG, payload)
    return_url = build_return_url(route, TOUR_CONFIG["survey_map"], payload, TOUR_CONFIG["task_name"])
    st.link_button("Continue to Survey", return_url, use_container_width=True)
else:
    st.caption("Complete all sections and click Get recommendation to continue.")