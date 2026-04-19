import time
from pathlib import Path
from urllib.parse import urlencode

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


VALID_GROUPS = {"visual", "text"}
VALID_APPS = {"app_a", "app_b", "app_c"}
VALID_STEPS = {"1", "2", "3"}


def hide_sidebar_nav():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def q(name: str) -> str:
    qp = st.query_params
    value = qp.get(name, "")
    if isinstance(value, list):
        return value[0] if value else ""
    return str(value).strip()


def get_route_value(name: str, default: str = "") -> str:
    value = st.session_state.get(name, "")
    if value is None or str(value).strip() == "":
        value = q(name)
    if value is None:
        return default
    return str(value).strip()


def validate_and_store_route():
    pid = get_route_value("pid")
    group = get_route_value("group")
    app1 = get_route_value("app1")
    app2 = get_route_value("app2")
    app3 = get_route_value("app3")
    step = get_route_value("step")
    app = get_route_value("app")

    errors = []
    if not pid:
        errors.append("Missing pid")
    if group not in VALID_GROUPS:
        errors.append("Invalid group")
    if app1 not in VALID_APPS:
        errors.append("Invalid app1")
    if app2 not in VALID_APPS:
        errors.append("Invalid app2")
    if app3 not in VALID_APPS:
        errors.append("Invalid app3")
    if step not in VALID_STEPS:
        errors.append("Invalid step")
    if app not in VALID_APPS:
        errors.append("Invalid app")

    expected_app = {"1": app1, "2": app2, "3": app3}.get(step)
    if expected_app and app != expected_app:
        errors.append(f"Expected app {expected_app} for step {step}, got {app}")

    if errors:
        st.error("Routing error. Please start from the Qualtrics entry link.")
        st.stop()

    route = {
        "pid": pid,
        "group": group,
        "app1": app1,
        "app2": app2,
        "app3": app3,
        "step": step,
        "app": app,
    }

    for k, v in route.items():
        st.session_state[k] = v

    return route


def _to_dense_1d(mat):
    if hasattr(mat, "toarray"):
        return np.asarray(mat.toarray()).ravel()
    return np.asarray(mat).ravel()


def base_feature_from_encoded_name(name: str, feature_group_map: dict) -> str:
    for prefix, label in feature_group_map.items():
        if name == prefix or name.startswith(prefix + "_"):
            return label
    return name


def aggregate_shap_to_study_features(shap_df: pd.DataFrame, feature_group_map: dict) -> pd.DataFrame:
    temp = shap_df.copy()
    temp["study_feature"] = temp["feature"].apply(lambda x: base_feature_from_encoded_name(x, feature_group_map))
    temp["abs_shap"] = temp["shap_value"].abs()
    out = (
        temp.groupby("study_feature", as_index=False)["abs_shap"]
        .sum()
        .rename(columns={"abs_shap": "importance"})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    out["xai_rank"] = np.arange(1, len(out) + 1)
    return out


def load_bundle(bundle_path: str):
    return joblib.load(bundle_path)


def compute_shap_for_row(bundle: dict, x_row: pd.DataFrame):
    pipe = bundle["model"]
    explainer = bundle["explainer"]
    feature_names = bundle["feature_names"]
    num_features = bundle["num_features"]

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    x_trans = pre.transform(x_row)
    x_vec = _to_dense_1d(x_trans)

    pred_class = pipe.predict(x_row)[0]
    class_idx = list(clf.classes_).index(pred_class)

    shap_values = explainer.shap_values(x_trans)
    base_values = explainer.expected_value

    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[class_idx]).ravel()
        bv = float(base_values[class_idx]) if hasattr(base_values, "__len__") else float(base_values)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            sv = arr[0, class_idx, :].ravel()
        else:
            sv = arr.ravel()
        bv = float(base_values[class_idx]) if hasattr(base_values, "__len__") else float(base_values)

    n = min(len(feature_names), len(x_vec), len(sv))
    df = pd.DataFrame({
        "feature": feature_names[:n],
        "value": x_vec[:n],
        "shap_value": sv[:n],
    })
    is_numeric = df["feature"].isin(num_features)
    active = (df["value"] != 0) | is_numeric
    df = df[active].copy()
    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns=["abs"])
    return pred_class, bv, df


def plot_shap_waterfall(shap_df: pd.DataFrame, base_value: float, max_display: int = 10):
    top = shap_df.head(max_display).copy()
    exp = shap.Explanation(
        values=top["shap_value"].values,
        base_values=base_value,
        data=top["value"].values,
        feature_names=top["feature"].tolist(),
    )
    plt.figure(figsize=(10, 5))
    shap.plots.waterfall(exp, show=False, max_display=max_display)
    plt.tight_layout()
    return plt.gcf()


def init_result_state(task_key: str):
    result_ready_key = f"{task_key}_result_ready"
    result_payload_key = f"{task_key}_result_payload"
    mm_rating_key = f"{task_key}_mental_model_ratings"

    if result_ready_key not in st.session_state:
        st.session_state[result_ready_key] = False
    if result_payload_key not in st.session_state:
        st.session_state[result_payload_key] = None

    return result_ready_key, result_payload_key, mm_rating_key


def render_mental_model_rating(feature_labels: list, state_key: str):
    if state_key not in st.session_state:
        st.session_state[state_key] = {}

    rating_options = {
        "Not important at all": 1,
        "Slightly important": 2,
        "Moderately important": 3,
        "Important": 4,
        "Very important": 5,
    }

    st.subheader("Before seeing the AI explanation")
    st.caption("Please rate how important each factor is, in your opinion, in influencing the AI’s recommendation.")

    ratings = {}
    all_answered = True

    for feature in feature_labels:
        key = f"{state_key}_{feature}"

        col1, col2 = st.columns([1.2, 3.2])

        with col1:
            st.markdown(
                f"""
                <div style="font-weight:600; font-size:15px; padding-top:8px;">
                    {feature} <span style="color:red">*</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            selected = st.radio(
                label=f"Rate {feature}",
                options=list(rating_options.keys()),
                index=None,
                horizontal=True,
                label_visibility="collapsed",
                key=key,
            )

        if selected is None:
            all_answered = False
        else:
            ratings[feature] = rating_options[selected]

        st.markdown("<div style='margin-bottom: 6px;'></div>", unsafe_allow_html=True)

    st.session_state[state_key] = ratings
    return ratings, all_answered


def build_return_url(route: dict, survey_map: dict, payload: dict, task_name: str):
    step = route["step"]
    if step not in survey_map:
        st.error("Invalid survey routing step.")
        st.stop()

    base_url = survey_map[step]
    params = {
        "pid": route["pid"],
        "group": route["group"],
        "app1": route["app1"],
        "app2": route["app2"],
        "app3": route["app3"],
        "step": route["step"],
        "current_app": route["app"],
        "task": task_name,
        "rec_id": payload["recommended_id"],
        "rec_name": payload["recommended_name"],
        "ts": payload["timestamp"],
    }

    for k, v in payload["inputs"].items():
        params[k] = v

    for k, v in payload["mental_model_ratings"].items():
        safe_key = k.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        params[f"mm_rating_{safe_key}"] = v

    for i, item in enumerate(payload["xai_rank_list"], start=1):
        params[f"xai_rank_{i}"] = item

    return f"{base_url}?{urlencode(params)}"


def _top_features(payload: dict, n: int = 3):
    return payload["xai_agg"]["study_feature"].tolist()[:n]


def _render_result_card(payload: dict, config: dict):
    st.subheader(config["result_title"])
    st.success(config["result_formatter"](payload))


def _render_visual_explanation(config: dict, payload: dict):
    st.subheader("Why this recommendation was made")
    st.caption(config["visual_caption"])

    top_features = _top_features(payload, 3)
    if top_features:
        st.markdown("**Main factors the model relied on:**")
        for feature in top_features:
            st.markdown(f"- **{feature}**")

    fig = plot_shap_waterfall(
        payload["shap_df"],
        base_value=payload["base_value"],
        max_display=config["max_shap_display"],
    )
    st.pyplot(fig)


def _render_text_explanation(config: dict, payload: dict):
    st.subheader("Why this recommendation was made")
    st.caption(config["text_caption"])

    builder = config.get("text_reason_builder")
    reasons = builder(payload) if callable(builder) else []

    if reasons:
        st.markdown("**Main reasons for this recommendation:**")
        for reason in reasons[:3]:
            st.markdown(f"- {reason}")
    else:
        top_features = _top_features(payload, 3)
        if len(top_features) >= 3:
            st.markdown(
                f"- The model mainly relied on **{top_features[0]}**, **{top_features[1]}**, and **{top_features[2]}**."
            )
        elif top_features:
            st.markdown(f"- The model mainly relied on **{', '.join(top_features)}**.")
        else:
            st.markdown("- The model relied on the most influential inputs in your response.")


def render_generic_result(route: dict, config: dict, payload: dict):
    group = route["group"]

    _render_result_card(payload, config)

    if group == "visual":
        _render_visual_explanation(config, payload)
    else:
        _render_text_explanation(config, payload)


def timestamp_now() -> int:
    return int(time.time())


def root_path() -> Path:
    return Path(__file__).resolve().parent