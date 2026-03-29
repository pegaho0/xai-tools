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
from configs.house_config import HOUSE_CONFIG

st.set_page_config(page_title="House Recommendation Study", layout="centered")
hide_sidebar_nav()
route = validate_and_store_route()

bundle = load_bundle(HOUSE_CONFIG["bundle_path"])
catalog = pd.read_csv("data/house_catalog.csv")
house_id_to_meta = {row["house_id"]: row.to_dict() for _, row in catalog.iterrows()}

result_ready_key, result_payload_key, mm_order_key = init_result_state("house")

st.title("🏠 House Recommendation")
st.caption(f"{'Visual Explanation' if route['group'] == 'visual' else 'Text Explanation'} • Step {route['step']} of 3")
st.caption(
    "Provide your housing preferences, rank the factors you think matter most to the AI, "
    "then get a house recommendation."
)

budget = st.slider("Budget (CAD)", 150000, 1500000, 550000, 10000)
city = st.selectbox("Preferred city", ["Montreal", "Quebec City", "Toronto", "Vancouver", "Calgary"], index=0)
property_type = st.selectbox("Property type", ["Condo", "Townhouse", "Detached house", "Semi-detached"], index=0)
bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=1)
bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
area_size = st.slider("Minimum preferred area size (sq ft)", 500, 4000, 1200, 50)
distance_to_downtown = st.selectbox("Preferred distance to downtown", ["Very close", "Close", "Moderate", "Far"], index=1)
public_transport_access = st.selectbox("Public transport access", ["Low", "Medium", "High"], index=2)
school_quality = st.selectbox("School quality importance", ["Low", "Medium", "High"], index=1)
safety = st.selectbox("Safety importance", ["Low", "Medium", "High", "Very high"], index=2)
noise_level = st.selectbox("Noise tolerance", ["Low", "Medium", "High"], index=1)
parking = st.selectbox("Need parking?", ["Yes", "No"], index=0)
garden = st.selectbox("Need garden or yard?", ["Yes", "No"], index=1)
view_quality = st.selectbox("Preferred view quality", ["Basic", "Good", "Excellent"], index=1)
building_age = st.selectbox("Preferred building age", ["New", "Moderate", "Older"], index=1)
investment_potential = st.selectbox("Investment potential importance", ["Low", "Medium", "High"], index=1)
property_tax_sensitivity = st.selectbox("Sensitivity to property tax", ["Low", "Medium", "High"], index=1)
family_suitability = st.selectbox("Family suitability importance", ["Low", "Medium", "High"], index=1)

x = pd.DataFrame([{
    "budget": budget,
    "city": city,
    "property_type": property_type,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "area_size": area_size,
    "distance_to_downtown": distance_to_downtown,
    "public_transport_access": public_transport_access,
    "school_quality": school_quality,
    "safety": safety,
    "noise_level": noise_level,
    "parking": parking,
    "garden": garden,
    "view_quality": view_quality,
    "building_age": building_age,
    "investment_potential": investment_potential,
    "property_tax_sensitivity": property_tax_sensitivity,
    "family_suitability": family_suitability,
}])

mental_model_order = render_mental_model_sort(HOUSE_CONFIG["mental_model_features"], mm_order_key)
mental_model_ranks = ranking_list_to_rank_dict(mental_model_order)

if st.button("Get recommendation", type="primary", use_container_width=True):
    pred_id = bundle["model"].predict(x)[0]
    meta = house_id_to_meta[pred_id]

    _, base_value, shap_df = compute_shap_for_row(bundle, x)
    xai_agg = aggregate_shap_to_study_features(shap_df, HOUSE_CONFIG["feature_group_map"])
    xai_rank_list = xai_agg["study_feature"].tolist()

    st.session_state[result_ready_key] = True
    st.session_state[result_payload_key] = {
        "timestamp": timestamp_now(),
        "inputs": {
            "budget": budget,
            "city": city,
            "property_type": property_type,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "area_size": area_size,
            "distance_to_downtown": distance_to_downtown,
            "public_transport_access": public_transport_access,
            "school_quality": school_quality,
            "safety": safety,
            "noise_level": noise_level,
            "parking": parking,
            "garden": garden,
            "view_quality": view_quality,
            "building_age": building_age,
            "investment_potential": investment_potential,
            "property_tax_sensitivity": property_tax_sensitivity,
            "family_suitability": family_suitability,
        },
        "mental_model_order": mental_model_order.copy(),
        "mental_model_ranks": mental_model_ranks,
        "recommended_id": pred_id,
        "recommended_name": meta["listing_name"],
        "meta": meta,
        "shap_df": shap_df,
        "base_value": base_value,
        "xai_agg": xai_agg,
        "xai_rank_list": xai_rank_list,
    }
    st.rerun()

if st.session_state[result_ready_key] and st.session_state[result_payload_key] is not None:
    payload = st.session_state[result_payload_key]
    render_generic_result(route, HOUSE_CONFIG, payload)
    return_url = build_return_url(route, HOUSE_CONFIG["survey_map"], payload, HOUSE_CONFIG["task_name"])
    st.link_button("Continue to Survey", return_url, use_container_width=True)
else:
    st.caption("Complete all sections and click Get recommendation to continue.")