import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st

from app_core import (
    aggregate_shap_to_study_features,
    build_return_url,
    compute_shap_for_row,
    hide_sidebar_nav,
    init_result_state,
    render_cad_text_input,
    render_choice_field,
    render_generic_result,
    render_mental_model_rating,
    timestamp_now,
    validate_and_store_route,
)
from configs.house_config import HOUSE_CONFIG

st.set_page_config(page_title="House Recommendation Study", layout="centered")
hide_sidebar_nav()
route = validate_and_store_route()

from model_loader import load_model_bundle
bundle = load_model_bundle("house")

catalog = pd.read_csv("data/house_catalog.csv")
house_id_to_meta = {row["house_id"]: row.to_dict() for _, row in catalog.iterrows()}

result_ready_key, result_payload_key, mm_rating_key = init_result_state("house")

st.title("🏠 House Recommendation")
st.caption(f"{'Visual Explanation' if route['group'] == 'visual' else 'Text Explanation'} • Step {route['step']} of 3")
st.caption(
    "Provide your housing preferences, rate how important you think each factor is to the AI, "
    "get a recommendation, and continue to the survey."
)

budget_text, budget = render_cad_text_input(
    "Budget (CAD) *",
    key="house_budget_text",
    placeholder="e.g. 650000 or CAD 650000",
)

city = render_choice_field(
    "Preferred city *",
    ["Montreal", "Quebec City", "Toronto", "Vancouver", "Calgary"],
    key="city",
    horizontal=False,
)

property_type = render_choice_field(
    "Property type *",
    ["Condo", "Townhouse", "Detached house", "Semi-detached"],
    key="property_type",
    horizontal=False,
)

bedrooms = render_choice_field(
    "Bedrooms *",
    [1, 2, 3, 4, 5],
    key="bedrooms",
    horizontal=False,
)

bathrooms = render_choice_field(
    "Bathrooms *",
    [1, 2, 3, 4],
    key="bathrooms",
    horizontal=False,
)

area_size = render_choice_field(
    "Minimum preferred area size (sq ft) *",
    list(range(500, 4001, 50)),
    key="area_size",
    horizontal=False,
)

distance_to_downtown = render_choice_field(
    "Preferred distance to downtown *",
    ["Very close", "Close", "Moderate", "Far"],
    key="distance_to_downtown",
    horizontal=False,
)

public_transport_access = render_choice_field(
    "Public transport access *",
    ["Low", "Medium", "High"],
    key="public_transport_access",
)

school_quality = render_choice_field(
    "School quality importance *",
    ["Low", "Medium", "High"],
    key="school_quality",
)

safety = render_choice_field(
    "Safety importance *",
    ["Low", "Medium", "High", "Very high"],
    key="safety",
    horizontal=False,
)

noise_level = render_choice_field(
    "Noise tolerance *",
    ["Low", "Medium", "High"],
    key="noise_level",
)

parking = render_choice_field(
    "Need parking? *",
    ["Yes", "No"],
    key="parking",
)

garden = render_choice_field(
    "Need garden or yard? *",
    ["Yes", "No"],
    key="garden",
)

view_quality = render_choice_field(
    "Preferred view quality *",
    ["Basic", "Good", "Excellent"],
    key="view_quality",
)

building_age = render_choice_field(
    "Preferred building age *",
    ["New", "Moderate", "Older"],
    key="building_age",
)

investment_potential = render_choice_field(
    "Investment potential importance *",
    ["Low", "Medium", "High"],
    key="investment_potential",
)

property_tax_sensitivity = render_choice_field(
    "Sensitivity to property tax *",
    ["Low", "Medium", "High"],
    key="property_tax_sensitivity",
)

family_suitability = render_choice_field(
    "Family suitability importance *",
    ["Low", "Medium", "High"],
    key="family_suitability",
)

mental_model_ratings, all_answered = render_mental_model_rating(
    HOUSE_CONFIG["mental_model_features"],
    mm_rating_key,
)

submit = st.button("Get recommendation", type="primary", use_container_width=True)

if submit:
    input_errors = []

    if budget is None:
        if str(budget_text).strip() == "":
            input_errors.append("Budget is required.")
        else:
            input_errors.append("Budget must be a valid amount in CAD.")
    if city is None:
        input_errors.append("Preferred city is required.")
    if property_type is None:
        input_errors.append("Property type is required.")
    if bedrooms is None:
        input_errors.append("Bedrooms is required.")
    if bathrooms is None:
        input_errors.append("Bathrooms is required.")
    if area_size is None:
        input_errors.append("Minimum preferred area size is required.")
    if distance_to_downtown is None:
        input_errors.append("Preferred distance to downtown is required.")
    if public_transport_access is None:
        input_errors.append("Public transport access is required.")
    if school_quality is None:
        input_errors.append("School quality importance is required.")
    if safety is None:
        input_errors.append("Safety importance is required.")
    if noise_level is None:
        input_errors.append("Noise tolerance is required.")
    if parking is None:
        input_errors.append("Parking preference is required.")
    if garden is None:
        input_errors.append("Garden or yard preference is required.")
    if view_quality is None:
        input_errors.append("Preferred view quality is required.")
    if building_age is None:
        input_errors.append("Preferred building age is required.")
    if investment_potential is None:
        input_errors.append("Investment potential importance is required.")
    if property_tax_sensitivity is None:
        input_errors.append("Property tax sensitivity is required.")
    if family_suitability is None:
        input_errors.append("Family suitability importance is required.")
    if not all_answered:
        input_errors.append("Please rate all factors before continuing.")

    if input_errors:
        for err in input_errors:
            st.error(err)
    else:
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
            "mental_model_ratings": mental_model_ratings.copy(),
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