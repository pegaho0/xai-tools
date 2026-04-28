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
    maybe_show_step1_welcome_modal,
    render_cad_text_input,
    render_choice_field,
    render_generic_result,
    render_mental_model_rating,
    timestamp_now,
    validate_and_store_route,
)
from configs.tour_config import TOUR_CONFIG

st.set_page_config(page_title="Tour Recommendation Study", layout="centered")
hide_sidebar_nav()
route = validate_and_store_route()
maybe_show_step1_welcome_modal(route)

from model_loader import load_model_bundle
bundle = load_model_bundle("tour")

catalog = pd.read_csv("data/tour_catalog.csv")
tour_id_to_meta = {row["tour_id"]: row.to_dict() for _, row in catalog.iterrows()}

result_ready_key, result_payload_key, mm_rating_key = init_result_state("tour")

st.title("🌍 Tour Recommendation")
st.caption(
    f"{'Visual Explanation' if route['group'] == 'visual' else 'Text Explanation'} • "
    f"Step {route['step']} of 3"
)
st.caption(
    "Provide your travel preferences, rate how important you think each factor is to the AI, "
    "then get a tour recommendation."
)

budget_text, budget = render_cad_text_input(
    "What is your budget (CAD) *",
    key="tour_budget_text",
    placeholder="e.g. 2500 or CAD 2500",
)

trip_duration = render_choice_field(
    "Preferred trip duration *",
    ["Short", "Medium", "Long"],
    key="trip_duration",
)

preferred_region = render_choice_field(
    "Preferred region *",
    ["Europe", "Asia", "North America", "South America", "Middle East", "Africa"],
    key="preferred_region",
    horizontal=False,
)

preferred_climate = render_choice_field(
    "Preferred climate *",
    ["Cold", "Mild", "Warm"],
    key="preferred_climate",
)

travel_style = render_choice_field(
    "Preferred travel style *",
    ["Adventure", "Relaxation", "Culture", "Nature", "Mixed"],
    key="travel_style",
    horizontal=False,
)

group_type = render_choice_field(
    "Who are you travelling with? *",
    ["Solo", "Couple", "Family", "Friends"],
    key="group_type",
    horizontal=False,
)

accommodation_level = render_choice_field(
    "Preferred accommodation level *",
    ["Budget", "Standard", "Premium", "Luxury"],
    key="accommodation_level",
    horizontal=False,
)

food_interest = render_choice_field(
    "How important is local food experience? *",
    ["Low", "Medium", "High"],
    key="food_interest",
)

transportation_comfort = render_choice_field(
    "Transportation comfort preference *",
    ["Basic", "Moderate", "High"],
    key="transportation_comfort",
)

season = render_choice_field(
    "Preferred season *",
    ["Spring", "Summer", "Autumn", "Winter"],
    key="season",
    horizontal=False,
)

safety_importance = render_choice_field(
    "How important is safety? *",
    ["Low", "Medium", "High", "Very high"],
    key="safety_importance",
    horizontal=False,
)

rating_importance = render_choice_field(
    "How important are tour ratings? *",
    ["Low", "Medium", "High", "Very high"],
    key="tour_rating_importance",
    horizontal=False,
)

mental_model_ratings, all_answered = render_mental_model_rating(
    TOUR_CONFIG["mental_model_features"],
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

    if trip_duration is None:
        input_errors.append("Trip duration is required.")
    if preferred_region is None:
        input_errors.append("Preferred region is required.")
    if preferred_climate is None:
        input_errors.append("Preferred climate is required.")
    if travel_style is None:
        input_errors.append("Travel style is required.")
    if group_type is None:
        input_errors.append("Group type is required.")
    if accommodation_level is None:
        input_errors.append("Accommodation level is required.")
    if food_interest is None:
        input_errors.append("Food interest is required.")
    if transportation_comfort is None:
        input_errors.append("Transportation comfort is required.")
    if season is None:
        input_errors.append("Season is required.")
    if safety_importance is None:
        input_errors.append("Safety importance is required.")
    if rating_importance is None:
        input_errors.append("Tour rating importance is required.")
    if not all_answered:
        input_errors.append("Please rate all factors before continuing.")

    if input_errors:
        for err in input_errors:
            st.error(err)
    else:
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

        pred_id = bundle["model"].predict(x)[0]
        meta = tour_id_to_meta[pred_id]

        _, base_value, shap_df = compute_shap_for_row(bundle, x)
        xai_agg = aggregate_shap_to_study_features(
            shap_df,
            TOUR_CONFIG["feature_group_map"],
        )
        xai_rank_list = xai_agg["study_feature"].tolist()

        st.session_state[result_ready_key] = True
        st.session_state[result_payload_key] = {
            "timestamp": timestamp_now(),
            "inputs": {
                "budget": budget,
                "budget_text": budget_text,
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
            "mental_model_ratings": mental_model_ratings.copy(),
            "recommended_id": pred_id,
            "recommended_name": meta["tour_name"],
            "meta": meta,
            "shap_df": shap_df,
            "base_value": base_value,
            "bundle": bundle,
            "x_row": x,
            "xai_agg": xai_agg,
            "xai_rank_list": xai_rank_list,
        }
        st.rerun()

if st.session_state[result_ready_key] and st.session_state[result_payload_key] is not None:
    payload = st.session_state[result_payload_key]
    render_generic_result(route, TOUR_CONFIG, payload)
    return_url = build_return_url(
        route,
        TOUR_CONFIG["survey_map"],
        payload,
        TOUR_CONFIG["task_name"],
    )
    st.link_button("Continue to Survey", return_url, use_container_width=True)
else:
    st.caption("Complete all sections and click Get recommendation to continue.")