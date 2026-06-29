import html
import re
import time
import textwrap
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components
from sklearn import tree


VALID_GROUPS = {"visual", "text"}
VALID_APPS = {"app_a", "app_b", "app_c"}
VALID_STEPS = {"1", "2", "3"}


def _render_html(html: str):
    """Render generated HTML without Markdown treating indented lines as code blocks."""
    # Streamlit uses Markdown rules before rendering HTML. If any generated
    # <div> line starts with 4+ spaces, Markdown displays it as a code block.
    # Therefore every line must be left-stripped, not only textwrap.dedent().
    cleaned = "\n".join(line.lstrip() for line in str(html).splitlines()).strip()
    st.markdown(cleaned, unsafe_allow_html=True)



def hide_sidebar_nav():
    st.markdown(
        """
        <style>

            /* Professional app layout */
            [data-testid="stSidebar"],
            [data-testid="stSidebarNav"],
            [data-testid="stSidebarCollapsedControl"] {
                display: none !important;
            }

            [data-testid="stAppViewContainer"] {
                background: #FFFFFF;
            }
            html, body, [data-testid="stAppViewContainer"] {
                color-scheme: light !important;
                background: #FFFFFF !important;
                color: #111827 !important;
            }

            /* Keep typography readable even if browser/extension forces dark mode */
            [data-testid="stAppViewContainer"] h1,
            [data-testid="stAppViewContainer"] h2,
            [data-testid="stAppViewContainer"] h3,
            [data-testid="stAppViewContainer"] p,
            [data-testid="stAppViewContainer"] label,
            [data-testid="stAppViewContainer"] span {
                color: #111827 !important;
            }

            /* Welcome modal polish */
            div[data-testid="stDialog"] {
                position: fixed !important;
                inset: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                padding: 24px !important;
                background: rgba(15, 23, 42, 0.38) !important;
                z-index: 99999 !important;
            }

            div[data-testid="stDialog"] > div {
                border-radius: 16px !important;
                border: 1px solid #E5E7EB !important;
                background: #FFFFFF !important;
                color: #111827 !important;
                width: min(720px, 92vw) !important;
                max-width: 720px !important;
                box-shadow: 0 20px 60px rgba(15, 23, 42, 0.30) !important;
                padding: 14px 18px 16px 18px !important;
            }

            /* Hide Streamlit dialog header title (top "Welcome") */
            div[data-testid="stDialog"] h1,
            div[data-testid="stDialog"] h2,
            div[data-testid="stDialog"] [data-testid="stHeading"] {
                display: none !important;
            }

            .welcome-modal-title {
                text-align: center;
                font-size: 32px;
                font-weight: 800;
                color: #111827 !important;
                margin: 6px 0 14px 0;
            }

            .welcome-modal-body {
                font-size: 20px;
                line-height: 1.7;
                color: #374151 !important;
                text-align: left;
                margin-bottom: 18px;
            }

            .block-container {
                max-width: 1120px;
                padding-top: 2.4rem;
                padding-left: 2.2rem;
                padding-right: 2.2rem;
            }

            /* Keep select boxes / expanders readable instead of stretched across the full page */
            div[data-baseweb="select"] {
                max-width: 640px;
            }

            div[data-testid="stExpander"] {
                max-width: 720px;
                border-radius: 12px;
            }

            div[data-testid="stForm"] {
                border: 0;
            }


            .mm-section-title {
                font-size: 15px;
                font-weight: 600;
                margin-top: 24px;
                margin-bottom: 8px;
            }

            .mm-section-subtitle {
                color: #555;
                font-size: 14px;
                margin-bottom: 14px;
            }

            .mm-feature-label {
                font-size: 14px;
                line-height: 1.35;
                color: #374151;
                padding-right: 10px;
                padding-top: 2px;
            }

            div[role="radiogroup"] label {
                margin-right: 6px !important;
                color: #111827 !important;
            }

            .cad-help {
                font-size: 12px;
                color: #666;
                margin-top: -8px;
                margin-bottom: 8px;
            }

            .xai-card {
                border: 1px solid #E5E7EB;
                border-radius: 14px;
                padding: 16px 18px;
                margin: 14px 0 18px 0;
                background: #FFFFFF;
                box-shadow: 0 1px 8px rgba(0,0,0,0.04);
            }

            .xai-small-note {
                color: #6B7280;
                font-size: 13px;
                line-height: 1.45;
                margin-top: -4px;
                margin-bottom: 12px;
            }

            .tree-path-intro {
                color: #4B5563;
                font-size: 14px;
                line-height: 1.5;
                margin-bottom: 12px;
            }

            .tree-rule-chip {
                display: inline-block;
                padding: 3px 8px;
                border-radius: 999px;
                background: #EEF2FF;
                color: #3730A3;
                font-weight: 700;
                font-size: 13px;
            }

            .tree-decision-box {
                border-left: 4px solid #6366F1;
                background: #F9FAFB;
                padding: 13px 15px;
                border-radius: 10px;
                margin-top: 8px;
                font-size: 15px;
                line-height: 1.6;
            }

            .tree-node-card {
                border: 1px solid #D1D5DB;
                border-radius: 14px;
                padding: 14px 15px;
                background: #FFFFFF;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                min-height: 150px;
            }

            .tree-node-title {
                font-size: 14px;
                font-weight: 800;
                color: #111827;
                margin-bottom: 6px;
            }

            .tree-node-rule {
                font-size: 14px;
                color: #374151;
                line-height: 1.45;
            }

            .tree-node-path {
                display: inline-block;
                margin-top: 8px;
                padding: 3px 8px;
                border-radius: 999px;
                background: #DCFCE7;
                color: #166534;
                font-size: 12px;
                font-weight: 700;
            }

            .tree-node-not-path {
                display: inline-block;
                margin-top: 8px;
                padding: 3px 8px;
                border-radius: 999px;
                background: #F3F4F6;
                color: #6B7280;
                font-size: 12px;
                font-weight: 700;
            }

            /* Unified form width and centered layout */
            .block-container {
                max-width: 980px !important;
                margin-left: auto !important;
                margin-right: auto !important;
                padding-top: 2.2rem !important;
                padding-left: 2.4rem !important;
                padding-right: 2.4rem !important;
            }

            div[data-testid="stTextInput"],
            div[data-testid="stSelectbox"],
            div[data-testid="stRadio"],
            div[data-testid="stButton"] {
                max-width: 860px !important;
            }

            div[data-baseweb="input"],
            div[data-baseweb="select"] {
                max-width: 860px !important;
                width: 100% !important;
            }

            div[data-baseweb="input"] input {
                background: #F3F4F6 !important;
                color: #111827 !important;
                border-color: #D1D5DB !important;
            }

            div[data-baseweb="input"] input::placeholder {
                color: #6B7280 !important;
                opacity: 1 !important;
            }

            div[data-baseweb="select"] > div {
                background: #F3F4F6 !important;
                color: #111827 !important;
                border-color: #D1D5DB !important;
            }

            /* Radio visibility in forced dark mode */
            div[data-testid="stRadio"] input[type="radio"] {
                accent-color: #2563EB !important;
            }

            .stButton > button {
                width: 100% !important;
                max-width: 860px !important;
                height: 48px !important;
                border-radius: 10px !important;
                font-size: 16px !important;
                font-weight: 700 !important;
                margin-top: 10px !important;
            }

            .mm-section-title {
                font-size: 16px !important;
                font-weight: 750 !important;
                margin-top: 26px !important;
                margin-bottom: 8px !important;
            }

            .mm-section-subtitle {
                color: #4B5563 !important;
                font-size: 14px !important;
                margin-bottom: 18px !important;
                max-width: 860px !important;
            }

            .mm-feature-label {
                font-size: 14px !important;
                line-height: 1.35 !important;
                color: #111827 !important;
                padding-top: 5px !important;
                padding-right: 8px !important;
                white-space: nowrap !important;
            }

            div[role="radiogroup"] {
                gap: 4px !important;
                align-items: center !important;
            }

            div[role="radiogroup"] label {
                margin-right: 2px !important;
                white-space: nowrap !important;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                max-width: 980px !important;
            }


            /* --- Clean unified question typography --- */
            .form-label {
                font-size: 15px !important;
                font-weight: 650 !important;
                color: #111827 !important;
                line-height: 1.35 !important;
                margin-bottom: 6px !important;
                white-space: normal !important;
                overflow-wrap: anywhere !important;
            }

            .inline-choice-label {
                font-size: 15px !important;
                font-weight: 650 !important;
                color: #111827 !important;
                line-height: 1.35 !important;
                padding-top: 6px !important;
                padding-right: 10px !important;
                /* Keep label + radios on one row; do not break the question mid-line */
                white-space: nowrap !important;
            }

            div[role="radiogroup"] {
                display: flex !important;
                flex-direction: row !important;
                flex-wrap: nowrap !important;
                align-items: center !important;
                justify-content: flex-start !important;
                gap: 7px !important;
                width: auto !important;
                max-width: none !important;
            }

            div[role="radiogroup"] label {
                margin-right: 4px !important;
                white-space: nowrap !important;
            }

            div[data-testid="stTextInput"],
            div[data-testid="stSelectbox"],
            div[data-testid="stButton"] {
                max-width: 900px !important;
            }

            div[data-baseweb="input"],
            div[data-baseweb="select"] {
                width: 100% !important;
                max-width: 900px !important;
            }

            .stButton > button {
                width: 100% !important;
                max-width: 900px !important;
            }


            /* Keep inline radio choices close to the question instead of drifting to the right */
            div[data-testid="column"] div[data-testid="stRadio"] {
                max-width: fit-content !important;
                width: fit-content !important;
            }

            div[data-testid="column"] div[role="radiogroup"] {
                justify-content: flex-start !important;
                width: fit-content !important;
                max-width: fit-content !important;
            }

            .inline-choice-label {
                padding-top: 2px !important;
                margin-bottom: 0 !important;
            }



            /* Stronger readable instructional text */
            div[data-testid="stCaptionContainer"] p {
                font-size: 16px !important;
                font-weight: 650 !important;
                color: #334155 !important;
                line-height: 1.55 !important;
            }
            .mm-section-title {
                font-size: 18px !important;
                font-weight: 850 !important;
                color: #0F172A !important;
                margin-top: 28px !important;
                margin-bottom: 8px !important;
            }
            .mm-section-subtitle {
                font-size: 16px !important;
                font-weight: 650 !important;
                color: #334155 !important;
                line-height: 1.55 !important;
                max-width: 900px !important;
            }
            .mm-feature-label {
                font-size: 15.5px !important;
                font-weight: 700 !important;
                color: #0F172A !important;
            }

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


def maybe_show_step1_welcome_modal(route: dict):
    """
    Show a one-time welcome/instructions modal at the start of Step 1.
    This is keyed by pid so it won't reappear for the same participant.
    """
    if not isinstance(route, dict):
        return
    if str(route.get("step", "")).strip() != "1":
        return

    pid = str(route.get("pid", "")).strip() or "anon"
    seen_key = f"welcome_modal_seen_step1_{pid}"
    if st.session_state.get(seen_key, False):
        return

    def _welcome_content():
        st.markdown("<div class='welcome-modal-title'>Welcome to our experiment!</div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class='welcome-modal-body'>
                Thank you for participating in this research.<br><br>
                This study has three steps. In each step, there is one application
                that gives you some recommendations.<br><br>
                After using each application, there is a short survey about that application.<br><br>
                Please complete the survey after each step and make sure to complete all three steps carefully.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("OK", type="primary", use_container_width=True):
            st.session_state[seen_key] = True
            st.rerun()

    # Streamlit Cloud/runtime differences can make st.dialog unavailable or invalid
    # in some contexts. Fallback keeps the experiment usable instead of crashing.
    try:
        # Streamlit requires a non-empty title; a single space keeps header text invisible.
        dialog_decorator = st.dialog(" ")
        dialog_decorator(_welcome_content)()
    except Exception:
        st.markdown(
            """
            <div style="border:1px solid #E5E7EB; border-radius:14px; padding:16px 18px; background:#FFFFFF; margin-bottom:14px;">
            """,
            unsafe_allow_html=True,
        )
        _welcome_content()
        st.markdown("</div>", unsafe_allow_html=True)


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
        temp.groupby("study_feature", as_index=False)
        .agg(
            importance=("abs_shap", "sum"),
            signed_effect=("shap_value", "sum"),
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    out["direction"] = np.where(out["signed_effect"] >= 0, "push_toward", "push_away")
    out["xai_rank"] = np.arange(1, len(out) + 1)
    return out


def load_bundle(bundle_path: str):
    return joblib.load(bundle_path)


def compute_shap_for_row(bundle: dict, x_row: pd.DataFrame):
    pipe = bundle["model"]
    explainer = bundle["explainer"]
    feature_names = bundle["feature_names"]

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    x_trans = pre.transform(x_row)

    if hasattr(x_trans, "toarray"):
        x_dense = x_trans.toarray().astype(float)
    else:
        x_dense = np.asarray(x_trans, dtype=float)

    x_vec = x_dense.ravel()

    pred_class = pipe.predict(x_row)[0]
    class_idx = list(clf.classes_).index(pred_class)

    shap_values = explainer.shap_values(x_dense)
    base_values = explainer.expected_value

    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[class_idx]).ravel()
        if hasattr(base_values, "__len__"):
            bv = float(base_values[class_idx])
        else:
            bv = float(base_values)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            sv = arr[0, :, class_idx]
        elif arr.ndim == 2:
            sv = arr[0, :]
        else:
            sv = arr.ravel()

        if hasattr(base_values, "__len__"):
            if len(np.asarray(base_values).shape) > 0 and len(base_values) > class_idx:
                bv = float(base_values[class_idx])
            else:
                bv = float(np.asarray(base_values).ravel()[0])
        else:
            bv = float(base_values)

    n = min(len(feature_names), len(x_vec), len(sv))
    df = pd.DataFrame({
        "feature": feature_names[:n],
        "value": x_vec[:n],
        "shap_value": sv[:n],
    })

    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns=["abs"]).reset_index(drop=True)
    return pred_class, bv, df


def parse_cad_input(value: str):
    """
    Accepts inputs like:
    1200
    1,200
    $1,200
    CAD 1200
    1200 CAD

    Returns int or None if invalid/empty.
    """
    if value is None:
        return None

    raw = str(value).strip()
    if raw == "":
        return None

    cleaned = raw.upper()
    cleaned = cleaned.replace("CAD", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.strip()

    if not re.fullmatch(r"\d+(\.\d+)?", cleaned):
        return None

    amount = float(cleaned)
    if amount < 0:
        return None

    return int(round(amount))


def render_cad_text_input(label: str, key: str, placeholder: str = "Enter amount in CAD"):
    st.markdown(
        f"<div class='form-label'>{label}</div>",
        unsafe_allow_html=True,
    )
    value = st.text_input(
        label="",
        key=key,
        placeholder=placeholder,
        label_visibility="collapsed",
    )
    st.markdown("<div class='cad-help'>Only Canadian dollars (CAD).</div>", unsafe_allow_html=True)
    parsed = parse_cad_input(value)
    return value, parsed

def render_choice_field(label: str, options: list, key: str, horizontal: bool = True):
    """
    Render every question label with the same custom font.
    For short option lists, label and radio buttons stay on the same row.
    For longer lists, a clean selectbox is shown under the label.
    """
    if len(options) <= 3:
        cols = st.columns([3.5, 6.5], gap="small")

        with cols[0]:
            st.markdown(
                f"<div class='inline-choice-label'>{label}</div>",
                unsafe_allow_html=True,
            )

        with cols[1]:
            return st.radio(
                label="",
                options=options,
                index=None,
                horizontal=True,
                key=key,
                label_visibility="collapsed",
            )

    st.markdown(
        f"<div class='form-label'>{label}</div>",
        unsafe_allow_html=True,
    )
    return st.selectbox(
        label="",
        options=options,
        index=None,
        placeholder="Choose an option",
        key=key,
        label_visibility="collapsed",
    )

def _extract_class_shap_matrix(shap_values, class_idx: int) -> np.ndarray:
    """Return SHAP values as (n_rows, n_features) for the selected class."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_idx], dtype=float)

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 3:
        return arr[:, :, class_idx]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    raise ValueError(f"Unsupported SHAP value shape: {arr.shape}")


def _aggregate_matrix_to_study_features(values_matrix: np.ndarray, feature_names: list, feature_group_map: dict) -> pd.DataFrame:
    """Aggregate encoded/OHE columns back to the original study-level features."""
    rows = []
    n_features = min(values_matrix.shape[1], len(feature_names))
    for j in range(n_features):
        rows.append({
            "encoded_feature": feature_names[j],
            "study_feature": base_feature_from_encoded_name(feature_names[j], feature_group_map),
            "j": j,
        })

    fmap = pd.DataFrame(rows)
    out = {}
    for study_feature, g in fmap.groupby("study_feature", sort=False):
        idx = g["j"].to_numpy()
        out[study_feature] = values_matrix[:, idx].sum(axis=1)

    return pd.DataFrame(out)


def _get_selected_class_index(bundle: dict, recommended_id):
    clf = bundle["model"].named_steps["clf"]
    classes = list(clf.classes_)
    return classes.index(recommended_id) if recommended_id in classes else 0


def _safe_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if np.isclose(vmin, vmax):
        return np.full_like(values, 0.5, dtype=float)
    return (values - vmin) / (vmax - vmin)


def plot_tree_shap_summary_like_reference(bundle: dict, payload: dict, feature_group_map: dict):
    """
    Paper-style TreeSHAP UI with improved readability:
    - LEFT: global feature importance bar chart.
    - RIGHT: SHAP beeswarm summary for all study-level features.
    - Colorbar is attached to the right of the beeswarm.
    """
    if bundle is None or "background_data" not in bundle:
        return None

    explainer = bundle["explainer"]
    feature_names = bundle["feature_names"]
    X_bg = np.asarray(bundle["background_data"], dtype=float)

    if X_bg.ndim != 2 or X_bg.shape[0] == 0:
        return None

    class_idx = _get_selected_class_index(bundle, payload.get("recommended_id"))
    bg_shap_raw = explainer.shap_values(X_bg)
    bg_shap = _extract_class_shap_matrix(bg_shap_raw, class_idx)

    n_cols = min(bg_shap.shape[1], X_bg.shape[1], len(feature_names))
    bg_shap = bg_shap[:, :n_cols]
    X_bg = X_bg[:, :n_cols]
    feature_names = feature_names[:n_cols]

    shap_grouped = _aggregate_matrix_to_study_features(bg_shap, feature_names, feature_group_map)
    value_grouped = _aggregate_matrix_to_study_features(X_bg, feature_names, feature_group_map)

    if shap_grouped.empty:
        return None

    ordered = shap_grouped.abs().mean(axis=0).sort_values(ascending=False).index.tolist()
    shap_grouped = shap_grouped[ordered]
    value_grouped = value_grouped[ordered]
    mean_abs = shap_grouped.abs().mean(axis=0)

    n_features = len(ordered)
    fig_height = max(7.2, 0.68 * n_features + 2.6)

    # Keep this figure moderate in pixel width. Huge matplotlib figures get
    # downscaled by Streamlit and make the text look tiny.
    fig = plt.figure(figsize=(14.2, fig_height), dpi=140)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=4,
        width_ratios=[1.08, 1.16, 0.22, 0.06],
        wspace=0.20,
    )
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_swarm = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 3])

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "xai_blue_red",
        ["#1E63FF", "#6EA8FF", "#D47AFF", "#FF0051"],
    )

    y_positions = np.arange(n_features)
    rng = np.random.default_rng(123)

    all_shap_abs_max = float(np.nanmax(np.abs(shap_grouped.to_numpy(dtype=float))))
    if not np.isfinite(all_shap_abs_max) or all_shap_abs_max <= 0:
        all_shap_abs_max = 1.0

    # LEFT: global feature importance bar chart.
    ax_bar.barh(
        y_positions,
        mean_abs.values,
        color="#1689E8",
        height=0.70,
        edgecolor="#1689E8",
        alpha=1.0,
    )
    ax_bar.set_yticks(y_positions)
    ax_bar.set_yticklabels(ordered, fontsize=15)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("mean(|SHAP value|)\n(average impact magnitude)", fontsize=14, labelpad=11)
    ax_bar.set_title("Mean |SHAP value|", fontsize=17, fontweight="bold", pad=14)
    ax_bar.grid(axis="x", color="#D7D7D7", linestyle="-", linewidth=0.9, alpha=0.8)
    ax_bar.grid(axis="y", color="#ECECEC", linestyle="-", linewidth=0.7, alpha=0.85)
    ax_bar.set_axisbelow(True)
    ax_bar.tick_params(axis="x", labelsize=12, pad=5)
    ax_bar.tick_params(axis="y", length=0, pad=7)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color("#555555")
    ax_bar.spines["bottom"].set_color("#555555")
    max_bar = float(mean_abs.max()) if len(mean_abs) else 0.0
    if max_bar > 0:
        ax_bar.set_xlim(0, max_bar * 1.14)

    # RIGHT: SHAP beeswarm / local explanation summary.
    for i, feature in enumerate(ordered):
        sv = shap_grouped[feature].to_numpy(dtype=float)
        vals = value_grouped[feature].to_numpy(dtype=float)
        norm_vals = _safe_minmax(vals)

        jitter = rng.normal(0, 0.078, size=len(sv))
        jitter = np.clip(jitter, -0.24, 0.24)
        y = np.full(len(sv), i, dtype=float) + jitter

        ax_swarm.scatter(
            sv,
            y,
            c=norm_vals,
            cmap=cmap,
            s=30,
            alpha=0.94,
            linewidths=0.10,
            edgecolors="white",
            rasterized=True,
        )

    ax_swarm.axvline(0, color="#4A4A4A", linewidth=1.05, alpha=0.95)
    ax_swarm.set_yticks(y_positions)
    ax_swarm.set_yticklabels([])
    ax_swarm.invert_yaxis()
    ax_swarm.set_xlabel("SHAP value (impact on model output)", fontsize=14, labelpad=11)
    ax_swarm.set_title("SHAP Summary – All Features", fontsize=17, fontweight="bold", pad=14)
    ax_swarm.grid(axis="x", color="#D7D7D7", linestyle="-", linewidth=0.9, alpha=0.8)
    ax_swarm.grid(axis="y", color="#ECECEC", linestyle="-", linewidth=0.7, alpha=0.85)
    ax_swarm.tick_params(axis="x", labelsize=12, pad=5)
    ax_swarm.tick_params(axis="y", labelleft=False, labelright=False, length=0, pad=0)
    ax_swarm.spines["top"].set_visible(False)
    ax_swarm.spines["right"].set_visible(False)
    ax_swarm.spines["left"].set_color("#555555")
    ax_swarm.spines["bottom"].set_color("#555555")
    ax_swarm.set_xlim(-all_shap_abs_max * 1.22, all_shap_abs_max * 1.22)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("Feature value", rotation=270, labelpad=20, fontsize=12)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=11, length=0, pad=6)
    cbar.outline.set_visible(False)

    fig.subplots_adjust(left=0.10, right=0.985, top=0.88, bottom=0.16)
    return fig

def get_tree_model_from_bundle(bundle: dict):
    if bundle is not None and bundle.get("surrogate_tree") is not None:
        return bundle["surrogate_tree"]

    clf = bundle["model"].named_steps["clf"]
    if hasattr(clf, "estimators_"):
        return clf.estimators_[0]
    return clf



def _tree_prediction_label(clf, node_id: int):
    """Return the majority class shown at a node."""
    values = np.asarray(clf.tree_.value[node_id][0])
    class_idx = int(np.argmax(values))
    if hasattr(clf, "classes_") and class_idx < len(clf.classes_):
        return str(clf.classes_[class_idx])
    return str(class_idx)


def _get_tree_path_context(payload: dict, config: dict):
    """Build all information needed to draw an interactive tree path."""
    bundle = payload.get("bundle")
    x_row = payload.get("x_row")
    if bundle is None or x_row is None:
        return None

    pipe = bundle["model"]
    pre = pipe.named_steps["pre"]
    clf = get_tree_model_from_bundle(bundle)
    if not hasattr(clf, "tree_") or not hasattr(clf, "decision_path"):
        return None

    feature_names = bundle.get("feature_names", [])
    feature_group_map = config.get("feature_group_map", {})
    x_trans = pre.transform(x_row)
    x_dense = x_trans.toarray() if hasattr(x_trans, "toarray") else np.asarray(x_trans)
    x_dense = np.asarray(x_dense, dtype=float)

    node_indicator = clf.decision_path(x_dense)
    path_nodes = node_indicator.indices[node_indicator.indptr[0]: node_indicator.indptr[1]].tolist()
    leaf_id = int(clf.apply(x_dense)[0])

    return {
        "bundle": bundle,
        "clf": clf,
        "tree": clf.tree_,
        "feature_names": feature_names,
        "feature_group_map": feature_group_map,
        "x_dense": x_dense,
        "path_nodes": path_nodes,
        "path_set": set(path_nodes),
        "leaf_id": leaf_id,
    }


def _tree_node_question(ctx: dict, node_id: int) -> str:
    tree_ = ctx["tree"]
    feature_names = ctx["feature_names"]
    feature_group_map = ctx["feature_group_map"]
    feature_idx = int(tree_.feature[node_id])

    if feature_idx < 0:
        return f"Final leaf → predicts { _tree_prediction_label(ctx['clf'], node_id) }"

    encoded_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
    base_label = base_feature_from_encoded_name(encoded_name, feature_group_map)
    threshold = float(tree_.threshold[node_id])

    for prefix, label in feature_group_map.items():
        if encoded_name.startswith(prefix + "_"):
            category = encoded_name[len(prefix) + 1:].replace("_", " ")
            return f"{label} = {category}?"

    return f"{base_label} ≤ {threshold:.2f}?"


def _tree_node_result_text(ctx: dict, node_id: int) -> str:
    tree_ = ctx["tree"]
    samples = int(tree_.n_node_samples[node_id]) if hasattr(tree_, "n_node_samples") else 0
    pred = _tree_prediction_label(ctx["clf"], node_id)
    if int(tree_.feature[node_id]) < 0:
        return f"Leaf node · prediction: {pred}"
    return f"Current likely output: {pred} · training cases: {samples}"


def _render_visual_node(ctx: dict, node_id: int, title: str, key_prefix: str, clickable: bool = False):
    """Draw one tree node as a readable visual card plus optional click button."""
    if node_id is None or node_id < 0:
        st.markdown(
            "<div class='tree-node-card' style='opacity:0.25; min-height:135px;'>No node</div>",
            unsafe_allow_html=True,
        )
        return False

    in_path = node_id in ctx["path_set"]
    is_leaf = node_id == ctx["leaf_id"]
    border = "#16A34A" if in_path else "#D1D5DB"
    bg = "#F0FDF4" if in_path else "#FFFFFF"
    badge_bg = "#DCFCE7" if in_path else "#F3F4F6"
    badge_color = "#166534" if in_path else "#6B7280"
    badge = "selected path" if in_path else "other branch"
    if is_leaf:
        badge = "final output"
        badge_bg = "#FEF3C7"
        badge_color = "#92400E"
        border = "#F59E0B"
        bg = "#FFFBEB"

    question = _tree_node_question(ctx, node_id)
    result_text = _tree_node_result_text(ctx, node_id)

    st.markdown(
        f"""
        <div class='tree-node-card' style='border:2px solid {border}; background:{bg}; min-height:145px;'>
            <div class='tree-node-title'>{title}</div>
            <div class='tree-node-rule' style='font-size:15px; font-weight:700;'>{question}</div>
            <div class='tree-node-rule' style='font-size:13px; margin-top:7px; color:#6B7280;'>{result_text}</div>
            <span style='display:inline-block; margin-top:9px; padding:4px 9px; border-radius:999px; background:{badge_bg}; color:{badge_color}; font-size:12px; font-weight:800;'>{badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if clickable and in_path and not is_leaf:
        return st.button("Open next level", key=f"{key_prefix}_{node_id}", use_container_width=True)
    return False


def render_clickable_visual_tree(payload: dict, config: dict):
    """Render a real tree-shaped, click-to-expand visual from root to selected leaf."""
    ctx = _get_tree_path_context(payload, config)
    if ctx is None:
        st.info("The visual tree could not be displayed. Re-train the model bundle with the surrogate tree included.")
        return

    path_nodes = ctx["path_nodes"]
    if not path_nodes:
        st.info("No decision path was found for this prediction.")
        return

    task_name = config.get("task_name", "task")
    rec_id = str(payload.get("recommended_id", "rec"))
    state_key = f"tree_visible_depth_{task_name}_{rec_id}"
    if state_key not in st.session_state:
        # 0 means show the top three nodes: root plus its left/right children.
        st.session_state[state_key] = 0

    max_depth_on_path = max(0, len(path_nodes) - 2)  # last item is leaf

    st.markdown("<div class='xai-card'>", unsafe_allow_html=True)
    st.markdown("**Clickable decision-tree surrogate**")
    st.markdown(
        "<div class='xai-small-note'>This is a real tree-shaped explanation. "
        "The green node is the branch followed for this participant. Click the green node to reveal the next level until the final leaf is reached.</div>",
        unsafe_allow_html=True,
    )

    if st.button("Reset tree", key=f"reset_{state_key}"):
        st.session_state[state_key] = 0
        st.rerun()

    visible_depth = int(st.session_state[state_key])
    visible_depth = min(visible_depth, max_depth_on_path)

    # Draw root row.
    center_cols = st.columns([1.15, 1.7, 1.15])
    with center_cols[1]:
        clicked = _render_visual_node(ctx, path_nodes[0], "Root node", f"open_{state_key}", clickable=False)

    # Top connector.
    st.markdown("<div style='text-align:center; font-size:24px; color:#9CA3AF; line-height:1;'>│<br>┴</div>", unsafe_allow_html=True)

    # For each visible depth, draw the two children of the current path node.
    for depth in range(0, visible_depth + 1):
        parent_id = path_nodes[depth]
        tree_ = ctx["tree"]
        left_id = int(tree_.children_left[parent_id]) if tree_.children_left[parent_id] != -1 else None
        right_id = int(tree_.children_right[parent_id]) if tree_.children_right[parent_id] != -1 else None
        next_path_node = path_nodes[depth + 1] if depth + 1 < len(path_nodes) else None

        st.markdown(
            f"<div style='text-align:center; color:#6B7280; font-weight:700; margin:8px 0 4px 0;'>Level {depth + 1}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='text-align:center; color:#9CA3AF; font-family:monospace; font-size:20px; margin-top:-8px;'>┌───────────────┴───────────────┐</div>",
            unsafe_allow_html=True,
        )

        cols = st.columns([1, 0.12, 1])
        clicked_left = clicked_right = False
        with cols[0]:
            clickable = left_id == next_path_node and depth == visible_depth and visible_depth < max_depth_on_path
            clicked_left = _render_visual_node(ctx, left_id, "NO / left branch", f"open_left_{state_key}_{depth}", clickable=clickable)
        with cols[2]:
            clickable = right_id == next_path_node and depth == visible_depth and visible_depth < max_depth_on_path
            clicked_right = _render_visual_node(ctx, right_id, "YES / right branch", f"open_right_{state_key}_{depth}", clickable=clickable)

        if clicked_left or clicked_right:
            st.session_state[state_key] = min(visible_depth + 1, max_depth_on_path)
            st.rerun()

        if depth < visible_depth:
            st.markdown("<div style='text-align:center; font-size:24px; color:#9CA3AF; line-height:1; margin:8px 0;'>│</div>", unsafe_allow_html=True)

    if visible_depth >= max_depth_on_path:
        st.success(f"Final selected option: {payload.get('recommended_name', payload.get('recommended_id', 'selected option'))}")
    else:
        st.caption("Click the green node on the selected path to reveal the next tree level.")

    st.markdown("</div>", unsafe_allow_html=True)




def _html_escape(value) -> str:
    """Small helper for safe HTML labels in Streamlit markdown blocks."""
    return html.escape("" if value is None else str(value))


def _importance_lookup_from_payload(payload: dict) -> dict:
    """Return {study_feature: importance row} from the already-computed SHAP aggregation."""
    xai_agg = payload.get("xai_agg")
    if xai_agg is None or not hasattr(xai_agg, "iterrows"):
        return {}
    lookup = {}
    for _, row in xai_agg.iterrows():
        feature = str(row.get("study_feature", "")).strip()
        if feature:
            lookup[feature] = {
                "importance": float(row.get("importance", 0.0) or 0.0),
                "signed_effect": float(row.get("signed_effect", 0.0) or 0.0),
                "direction": str(row.get("direction", "")),
                "rank": int(row.get("xai_rank", 0) or 0),
            }
    return lookup


def _tree_node_feature_label(ctx: dict, node_id: int) -> str:
    tree_ = ctx["tree"]
    feature_idx = int(tree_.feature[node_id])
    if feature_idx < 0:
        return "Final recommendation"
    feature_names = ctx.get("feature_names", [])
    feature_group_map = ctx.get("feature_group_map", {})
    encoded_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
    return base_feature_from_encoded_name(encoded_name, feature_group_map)


def _tree_node_branch_taken(ctx: dict, node_id: int, next_node_id: int) -> str:
    """Return a human-readable branch label for the child selected by the participant."""
    tree_ = ctx["tree"]
    if int(tree_.feature[node_id]) < 0:
        return "Final"
    left_id = int(tree_.children_left[node_id])
    right_id = int(tree_.children_right[node_id])
    if next_node_id == right_id:
        return "Yes branch"
    if next_node_id == left_id:
        return "No branch"
    return "Selected branch"


def _hybrid_path_steps(ctx: dict, payload: dict) -> list:
    """Create compact path steps: tree rule + matching SHAP importance."""
    importance_lookup = _importance_lookup_from_payload(payload)
    steps = []
    path_nodes = ctx.get("path_nodes", [])

    for pos, node_id in enumerate(path_nodes):
        is_leaf = node_id == ctx.get("leaf_id") or int(ctx["tree"].feature[node_id]) < 0
        feature_label = _tree_node_feature_label(ctx, node_id)
        imp = importance_lookup.get(feature_label, {})
        next_node = path_nodes[pos + 1] if pos + 1 < len(path_nodes) else None
        steps.append({
            "node_id": int(node_id),
            "position": pos + 1,
            "is_leaf": bool(is_leaf),
            "feature": feature_label,
            "question": _tree_node_question(ctx, node_id),
            "branch": _tree_node_branch_taken(ctx, node_id, next_node) if next_node is not None else "Final output",
            "prediction": _tree_prediction_label(ctx["clf"], node_id),
            "importance": float(imp.get("importance", 0.0) or 0.0),
            "signed_effect": float(imp.get("signed_effect", 0.0) or 0.0),
            "rank": int(imp.get("rank", 0) or 0),
        })
    return steps


def _render_hybrid_top_bars(payload: dict, max_items: int = 7):
    xai_agg = payload.get("xai_agg")
    if xai_agg is None or not hasattr(xai_agg, "head") or xai_agg.empty:
        st.info("SHAP importance values were not available for this explanation.")
        return

    top = xai_agg.head(max_items).copy()
    max_imp = float(top["importance"].max()) if "importance" in top else 0.0
    if max_imp <= 0:
        max_imp = 1.0

    rows_html = []
    for _, row in top.iterrows():
        feature = _html_escape(row.get("study_feature", "Feature"))
        imp = float(row.get("importance", 0.0) or 0.0)
        width = max(5, min(100, int(round((imp / max_imp) * 100))))
        direction = str(row.get("direction", ""))
        dir_label = "supports" if direction == "push_toward" else "weaker fit"
        pill_bg = "#DCFCE7" if direction == "push_toward" else "#FEE2E2"
        pill_color = "#166534" if direction == "push_toward" else "#991B1B"
        rows_html.append(f"""
        <div style='margin:10px 0 13px 0;'>
            <div style='display:flex; justify-content:space-between; align-items:center; gap:10px;'>
                <div style='font-size:13px; font-weight:750; color:#111827;'>{feature}</div>
                <div style='font-size:12px; font-weight:750; color:#374151;'>{imp:.3f}</div>
            </div>
            <div style='height:10px; border-radius:999px; background:#E5E7EB; overflow:hidden; margin-top:5px;'>
                <div style='height:10px; width:{width}%; border-radius:999px; background:linear-gradient(90deg,#2563EB,#7C3AED);'></div>
            </div>
            <span style='display:inline-block; margin-top:5px; padding:2px 7px; border-radius:999px; background:{pill_bg}; color:{pill_color}; font-size:11px; font-weight:800;'>{dir_label}</span>
        </div>
        """)

    _render_html("""
    <div style='font-size:15px; font-weight:850; color:#111827; margin-bottom:4px;'>Top SHAP factors</div>
    <div style='font-size:12.5px; color:#6B7280; line-height:1.45; margin-bottom:8px;'>
    Longer bars mean the factor had stronger influence on the recommendation.
    </div>
    """ + "".join(rows_html))


def _render_hybrid_path_cards(steps: list, recommended_name: str):
    if not steps:
        st.info("No tree path was available for this participant.")
        return

    max_imp = max([s.get("importance", 0.0) for s in steps] + [1e-9])
    cards = []
    for step in steps:
        if step.get("is_leaf"):
            cards.append(f"""
            <div style='border:2px solid #F59E0B; background:#FFFBEB; border-radius:16px; padding:14px 16px; margin:10px 0;'>
                <div style='font-size:12px; font-weight:850; color:#92400E; text-transform:uppercase;'>Final node</div>
                <div style='font-size:16px; font-weight:850; color:#111827; margin-top:4px;'>Recommended: {_html_escape(recommended_name)}</div>
                <div style='font-size:13px; color:#6B7280; margin-top:5px;'>The selected tree path ends here.</div>
            </div>
            """)
            continue

        imp = float(step.get("importance", 0.0) or 0.0)
        width = max(4, min(100, int(round((imp / max_imp) * 100))))
        branch = _html_escape(step.get("branch", "Selected branch"))
        question = _html_escape(step.get("question", ""))
        feature = _html_escape(step.get("feature", ""))
        rank = step.get("rank", 0)
        rank_text = f"Rank #{rank}" if rank else "Tree split feature"
        cards.append(f"""
        <div style='border:1px solid #D1D5DB; background:#FFFFFF; border-radius:16px; padding:14px 16px; margin:10px 0; box-shadow:0 2px 10px rgba(15,23,42,0.05);'>
            <div style='display:flex; justify-content:space-between; gap:10px; align-items:center;'>
                <div style='font-size:12px; font-weight:850; color:#4F46E5; text-transform:uppercase;'>Step {step.get('position')}</div>
                <span style='padding:3px 8px; border-radius:999px; background:#EEF2FF; color:#3730A3; font-size:11px; font-weight:850;'>{rank_text}</span>
            </div>
            <div style='font-size:15px; font-weight:850; color:#111827; margin-top:6px;'>{question}</div>
            <div style='font-size:13px; color:#4B5563; margin-top:6px;'>Selected path: <b>{branch}</b></div>
            <div style='margin-top:9px;'>
                <div style='display:flex; justify-content:space-between; font-size:12px; color:#6B7280;'>
                    <span>SHAP importance for {feature}</span><span>{imp:.3f}</span>
                </div>
                <div style='height:9px; border-radius:999px; background:#E5E7EB; overflow:hidden; margin-top:5px;'>
                    <div style='height:9px; width:{width}%; border-radius:999px; background:linear-gradient(90deg,#22C55E,#2563EB);'></div>
                </div>
            </div>
        </div>
        """)

    _render_html("""
    <div style='font-size:15px; font-weight:850; color:#111827; margin-bottom:4px;'>Your decision path</div>
    <div style='font-size:12.5px; color:#6B7280; line-height:1.45; margin-bottom:8px;'>
    The tree gives the readable route. SHAP shows how strong each route feature was.
    </div>
    """ + "".join(cards))


def render_hybrid_shap_tree_explanation(payload: dict, config: dict):
    """
    Third explanation: combines the decision-tree path with SHAP feature strength.
    This is intentionally user-facing: the tree provides structure, SHAP provides weight.
    """
    ctx = _get_tree_path_context(payload, config)
    if ctx is None:
        st.info("Hybrid SHAP + tree explanation could not be displayed. Re-train the bundle with a surrogate_tree and background_data.")
        return

    steps = _hybrid_path_steps(ctx, payload)
    recommended_name = payload.get("recommended_name", payload.get("recommended_id", "selected option"))
    top_features = payload.get("xai_agg")
    top_feature = "the strongest factors"
    if top_features is not None and hasattr(top_features, "empty") and not top_features.empty:
        top_feature = str(top_features.iloc[0].get("study_feature", top_feature))

    with st.container(border=True):
        st.markdown("### Hybrid SHAP + Decision Tree Explanation")
        st.markdown(
            """
            <div style='color:#4B5563; font-size:14px; line-height:1.55; margin-bottom:12px;'>
            This view combines two explanations: <b>the tree</b> shows the step-by-step logic, while <b>SHAP</b> shows how strongly each factor influenced the recommendation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        summary_cols = st.columns([1, 1, 1])
        with summary_cols[0]:
            st.markdown(
                f"""
                <div style='border:1px solid #E5E7EB; border-radius:14px; padding:13px 15px; background:#F9FAFB; min-height:92px;'>
                    <div style='font-size:12px; color:#6B7280; font-weight:800; text-transform:uppercase;'>Recommendation</div>
                    <div style='font-size:17px; color:#111827; font-weight:850; margin-top:5px;'>{_html_escape(recommended_name)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with summary_cols[1]:
            st.markdown(
                f"""
                <div style='border:1px solid #E5E7EB; border-radius:14px; padding:13px 15px; background:#F9FAFB; min-height:92px;'>
                    <div style='font-size:12px; color:#6B7280; font-weight:800; text-transform:uppercase;'>Strongest SHAP factor</div>
                    <div style='font-size:17px; color:#111827; font-weight:850; margin-top:5px;'>{_html_escape(top_feature)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with summary_cols[2]:
            n_steps = max(0, len([s for s in steps if not s.get("is_leaf")]))
            st.markdown(
                f"""
                <div style='border:1px solid #E5E7EB; border-radius:14px; padding:13px 15px; background:#F9FAFB; min-height:92px;'>
                    <div style='font-size:12px; color:#6B7280; font-weight:800; text-transform:uppercase;'>Tree path length</div>
                    <div style='font-size:17px; color:#111827; font-weight:850; margin-top:5px;'>{n_steps} decision steps</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        left, right = st.columns([1.25, 1.0], gap="large")
        with left:
            _render_hybrid_path_cards(steps, recommended_name)
        with right:
            _render_hybrid_top_bars(payload, max_items=7)

        st.markdown(
            """
            <div style='border-left:4px solid #2563EB; background:#EFF6FF; padding:12px 14px; border-radius:12px; margin-top:10px; color:#1E3A8A; font-size:13.5px; line-height:1.55;'>
            <b>How to read it:</b> follow the cards from top to bottom. Each card is one tree decision. The SHAP bar inside the card tells whether that decision feature was also important according to the model's feature-attribution analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )
def render_readable_decision_path(payload: dict, config: dict):
    render_clickable_visual_tree(payload, config)


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

    st.markdown(
        "<div class='mm-section-title'>Your Expectations Before the AI Explanation</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='mm-section-subtitle'>Before seeing the AI explanation, please rate how important you think each feature will be in the AI's decision, from 1 (not important at all) to 7 (extremely important).</div>",
        unsafe_allow_html=True,
    )

    ratings = {}
    all_answered = True

    for feature in feature_labels:
        safe_feature = (
            str(feature).lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )
        key = f"{state_key}_{safe_feature}"

        outer = st.columns([6, 6], gap="small")

        with outer[0]:
            st.markdown(
                f"<div class='mm-feature-label'>{feature} is important in the AI’s decision.</div>",
                unsafe_allow_html=True,
            )

        with outer[1]:
            selected = st.radio(
                label=f"{feature} - scale",
                options=[1, 2, 3, 4, 5, 6, 7],
                index=None,
                horizontal=True,
                key=key,
                label_visibility="collapsed",
            )

        if selected is None:
            all_answered = False
        else:
            ratings[feature] = selected

    st.session_state[state_key] = ratings
    return ratings, all_answered


def clean_name(s):
    """
    Normalize visible feature labels into the exact suffix used in Qualtrics.
    Example:
        'Dietary restriction / allergy' -> 'dietary_restriction_allergy'
        'Importance of customer rating' -> 'importance_of_customer_rating'
    """
    s = str(s).lower().strip()
    s = s.replace("&", "and")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def build_qualtrics_params(payload: dict, task_name: str) -> dict:
    """
    Build only the Qualtrics embedded-data parameters that must be saved.

    Output examples:
        pizza_mm_maximum_price=1
        pizza_xai_rank_1=Pizza style
        tour_mm_budget=2
        house_xai_rank_1=City
    """
    params = {}

    mental_model_name_map = {
        # Pizza
        "maximum_price": "maximum_price",
        "pizza_style": "pizza_style",
        "ingredient_preference": "ingredient_preference",
        "dietary_restriction_allergy": "dietary_restriction_allergy",
        "importance_of_customer_rating": "customer_rating",
        "customer_rating": "customer_rating",
        "importance_of_free_delivery": "free_delivery",
        "free_delivery": "free_delivery",

        # Tour
        "budget": "budget",
        "trip_duration": "trip_duration",
        "preferred_region": "preferred_region",
        "preferred_climate": "preferred_climate",
        "travel_style": "travel_style",
        "group_type": "group_type",
        "accommodation_level": "accommodation_level",
        "food_interest": "food_interest",
        "transportation_comfort": "transportation_comfort",
        "season": "season",
        "safety_importance": "safety_importance",
        "rating_importance": "rating_importance",

        # House
        "city": "city",
        "property_type": "property_type",
        "bedrooms": "bedrooms",
        "bathrooms": "bathrooms",
        "area_size": "area_size",
        "distance_to_downtown": "distance_to_downtown",
        "public_transport_access": "public_transport_access",
        "school_quality": "school_quality",
        "safety": "safety",
        "noise_level": "noise_level",
        "parking": "parking",
        "garden": "garden",
        "view_quality": "view_quality",
        "building_age": "building_age",
        "investment_potential": "investment_potential",
        "property_tax_sensitivity": "property_tax_sensitivity",
        "family_suitability": "family_suitability",
    }

    # Mental-model ratings
    for feature_label, rating in payload.get("mental_model_ratings", {}).items():
        suffix = mental_model_name_map.get(clean_name(feature_label))
        if suffix is not None and rating is not None:
            params[f"{task_name}_mm_{suffix}"] = rating

    # XAI / SHAP ranking
    for i, feature_label in enumerate(payload.get("xai_rank_list", []), start=1):
        if feature_label is not None:
            params[f"{task_name}_xai_rank_{i}"] = feature_label

    return params


def build_return_url(route: dict, survey_map: dict, payload: dict, task_name: str):
    """
    Build the Qualtrics survey URL after a Streamlit app is completed.

    Important:
    This function intentionally does NOT send the old names:
        mm_rating_...
        xai_rank_...

    It sends the exact names you defined in Qualtrics:
        pizza_mm_..., pizza_xai_rank_...
        tour_mm_..., tour_xai_rank_...
        house_mm_..., house_xai_rank_...

    ``AnalyzeTime`` (seconds on the explanation screen) is included in the query string when
    ``mark_analyze_phase_start`` has run; it is placed **early** (right after ``pid``) so very long
    URLs are less likely to lose it to length limits. ``render_timed_continue_to_survey`` refreshes
    the link about once per second so the value stays current.
    """
    step = str(route["step"]).strip()

    if step not in survey_map:
        st.error("Invalid survey routing step.")
        st.stop()

    base_url = survey_map[step]

    extra = build_qualtrics_params(payload, task_name)

    params = {"pid": route["pid"]}

    start_key = f"{task_name}_analyze_phase_start"
    if start_key in st.session_state:
        elapsed = max(0.0, time.time() - float(st.session_state[start_key]))
        params["AnalyzeTime"] = f"{elapsed:.2f}"

    params.update(
        {
            "group": route["group"],
            "app1": route["app1"],
            "app2": route["app2"],
            "app3": route["app3"],
            "step": route["step"],
            "current_app": route["app"],
            "task": task_name,
            "rec_id": payload.get("recommended_id", ""),
            "rec_name": payload.get("recommended_name", ""),
            "ts": payload.get("timestamp", ""),
        }
    )
    params.update(extra)

    return f"{base_url}?{urlencode(params)}"


def mark_analyze_phase_start(task_name: str) -> None:
    """Record when the recommendation is ready (after a successful *Get recommendation*). Used for Qualtrics ``AnalyzeTime``."""
    st.session_state[f"{task_name}_analyze_phase_start"] = time.time()


def render_timed_continue_to_survey(
    route: dict,
    survey_map: dict,
    payload: dict,
    task_name: str,
    label: str = "Continue to Survey",
) -> None:
    """
    Show **Continue to Survey** with a Qualtrics URL that includes ``AnalyzeTime`` (seconds).

    ``AnalyzeTime`` is set inside ``build_return_url`` (early in the query string). When
    ``st.fragment`` is available, the link is refreshed about once per second so the value
    stays close to the time spent on the explanation screen.
    """
    start_key = f"{task_name}_analyze_phase_start"
    if start_key not in st.session_state:
        st.session_state[start_key] = time.time()

    fragment_fn = getattr(st, "fragment", None)
    if callable(fragment_fn):
        @fragment_fn(run_every=1.0)
        def _refreshing_continue_link():
            st.link_button(
                label,
                build_return_url(route, survey_map, payload, task_name),
                use_container_width=True,
            )

        _refreshing_continue_link()
    else:
        st.link_button(
            label,
            build_return_url(route, survey_map, payload, task_name),
            use_container_width=True,
        )


def _top_features(payload: dict, n: int = None, min_n: int = 0):
    return payload["xai_agg"]["study_feature"].tolist()


def _render_result_card(payload: dict, config: dict):
    st.subheader(config["result_title"])
    st.success(config["result_formatter"](payload))


def _default_reason_builder(payload: dict):
    reasons = []
    top = payload["xai_agg"]

    for _, row in top.iterrows():
        feature = row["study_feature"]
        direction = row["direction"]
        if direction == "push_toward":
            reasons.append(f"{feature} supported this recommendation.")
        else:
            reasons.append(f"{feature} had a weaker fit, but the model still selected this option overall.")

    return reasons


def _render_visual_explanation(config: dict, payload: dict):
    st.subheader("Why this recommendation was made")
    st.caption(
        "This explanation uses a tree-based model and TreeSHAP values. "
        "All input features are included in the ranking."
    )

    # Use a native Streamlit bordered container instead of raw HTML <div>.
    # The previous HTML card created an empty blank section because Streamlit
    # elements do not reliably stay inside manually opened/closed HTML divs.
    with st.container(border=True):
        st.markdown("**SHAP feature-attribution summary**")
        st.markdown(
            """
            <div style="color:#6B7280; font-size:13px; line-height:1.55; margin-bottom:12px;">
                Left: how feature values push the model output across background cases.<br>
                Right: overall importance of each factor.
            </div>
            """,
            unsafe_allow_html=True,
        )

        shap_fig = plot_tree_shap_summary_like_reference(
            payload.get("bundle"),
            payload,
            config.get("feature_group_map", {}),
        )

        if shap_fig is not None:
            st.pyplot(shap_fig, use_container_width=True)
        else:
            st.warning(
                "TreeSHAP summary plot could not be generated. "
                "Check that the model bundle contains background_data."
            )

    render_clickable_visual_tree(payload, config)

    # Third explanation: combined tree structure + SHAP importance for easier comparison.
    render_hybrid_shap_tree_explanation(payload, config)

def _render_text_explanation(config: dict, payload: dict):
    """Textual XAI: same surrogate-tree path and SHAP semantics as the visual condition, text presentation only."""
    _inject_xai_dashboard_css()
    st.markdown("<div class='xai-dashboard-title'>How the model made this recommendation</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='xai-dashboard-subtitle'>First, carefully review the written explanation of how the recommendation was generated, paying attention to each step in the decision process. Then, read the SHAP explanation describing the most influential factors behind the recommendation. You will answer questions about these explanations in the next section.</div>",
        unsafe_allow_html=True,
    )

    ctx = _get_tree_path_context(payload, config)
    if ctx is not None:
        _render_path_walkthrough(ctx, payload)
    else:
        st.warning(
            "Textual decision-tree explanation could not be shown. "
            "Re-train the model bundle with surrogate_tree included."
        )

    st.markdown(
        '<p style="font-weight:850;font-size:1.2rem;color:#0F172A;margin:1.1rem 0 0.35rem 0;line-height:1.35;">'
        "Feature importance (SHAP)"
        "</p>",
        unsafe_allow_html=True,
    )
    st.caption(config["text_caption"])

    builder = config.get("text_reason_builder")
    reasons = builder(payload) if callable(builder) else _default_reason_builder(payload)

    if reasons:
        st.markdown("**Main reasons for this recommendation:**")
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        top_features = _top_features(
            payload,
            n=config.get("max_shap_display", 6),
            min_n=config.get("min_shap_display", 4),
        )
        if top_features:
            st.markdown(f"- The model mainly considered: **{', '.join(top_features)}**.")
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


# -----------------------------------------------------------------------------
# FINAL OVERRIDE: readable XAI dashboard with SHAP on top and full path-highlighted tree below
# -----------------------------------------------------------------------------

def _xai_html(value) -> str:
    return html.escape("" if value is None else str(value))


def _xai_prediction_label(ctx: dict, node_id: int) -> str:
    try:
        return _tree_prediction_label(ctx["clf"], int(node_id))
    except Exception:
        return "selected option"


def _xai_node_stats(ctx: dict, node_id: int) -> dict:
    tree_ = ctx["tree"]
    node_id = int(node_id)
    samples = int(tree_.n_node_samples[node_id]) if hasattr(tree_, "n_node_samples") else 0
    return {"samples": samples, "prediction": _xai_prediction_label(ctx, node_id)}


def _xai_split_details(ctx: dict, node_id: int) -> dict:
    """Return user-facing split details for a node."""
    tree_ = ctx["tree"]
    feature_idx = int(tree_.feature[node_id])
    if feature_idx < 0:
        stats = _xai_node_stats(ctx, node_id)
        return {
            "is_leaf": True,
            "feature": "Final recommendation",
            "question": f"Final recommendation: {stats['prediction']}",
            "condition": "The decision path ends here.",
            "left_label": "",
            "right_label": "",
        }

    feature_names = ctx.get("feature_names", [])
    feature_group_map = ctx.get("feature_group_map", {})
    encoded_name = feature_names[feature_idx] if feature_idx < len(feature_names) else f"Feature {feature_idx}"
    feature_label = base_feature_from_encoded_name(encoded_name, feature_group_map)
    threshold = float(tree_.threshold[node_id])

    # One-hot encoded category split: feature_category <= 0.5 means category is NOT selected.
    for prefix, label in feature_group_map.items():
        if encoded_name.startswith(prefix + "_"):
            category = encoded_name[len(prefix) + 1:].replace("_", " ")
            return {
                "is_leaf": False,
                "feature": label,
                "question": f"Is {label.lower()} '{category}'?",
                "condition": f"The tree checks whether the user's answer for {label.lower()} is '{category}'.",
                "left_label": "No",
                "right_label": "Yes",
            }

    return {
        "is_leaf": False,
        "feature": feature_label,
        "question": f"Is {feature_label.lower()} ≤ {threshold:.2f}?",
        "condition": f"The tree compares the user's {feature_label.lower()} with {threshold:.2f}.",
        "left_label": f"Yes, ≤ {threshold:.2f}",
        "right_label": f"No, > {threshold:.2f}",
    }


def _xai_branch_label(ctx: dict, node_id: int, child_id: int) -> str:
    details = _xai_split_details(ctx, node_id)
    tree_ = ctx["tree"]
    if int(tree_.feature[node_id]) < 0:
        return "Final"
    left_id = int(tree_.children_left[node_id])
    right_id = int(tree_.children_right[node_id])
    if child_id == left_id:
        return details.get("left_label", "Left branch")
    if child_id == right_id:
        return details.get("right_label", "Right branch")
    return "Selected branch"


def _xai_path_steps(ctx: dict, payload: dict) -> list:
    path_nodes = ctx.get("path_nodes", [])
    steps = []
    for idx, node_id in enumerate(path_nodes):
        node_id = int(node_id)
        next_node = int(path_nodes[idx + 1]) if idx + 1 < len(path_nodes) else None
        details = _xai_split_details(ctx, node_id)
        stats = _xai_node_stats(ctx, node_id)
        steps.append({
            "node_id": node_id,
            "step": idx + 1,
            "is_leaf": details["is_leaf"],
            "feature": details["feature"],
            "question": details["question"],
            "condition": details["condition"],
            "branch": _xai_branch_label(ctx, node_id, next_node) if next_node is not None else "Final output",
            "samples": stats["samples"],
            "prediction": str(payload.get("recommended_name", payload.get("recommended_id", stats["prediction"]))),
        })
    return steps


def _render_readable_shap_card(payload: dict, max_items: int = 8):
    xai_agg = payload.get("xai_agg")
    if xai_agg is None or not hasattr(xai_agg, "head") or xai_agg.empty:
        st.info("SHAP importance values were not available for this explanation.")
        return

    top = xai_agg.head(max_items).copy()
    max_imp = float(top["importance"].max()) if "importance" in top else 0.0
    if max_imp <= 0:
        max_imp = 1.0

    rows = []
    for _, row in top.iterrows():
        feature = _xai_html(row.get("study_feature", "Feature"))
        imp = float(row.get("importance", 0.0) or 0.0)
        width = max(3, min(100, int(round((imp / max_imp) * 100))))
        direction = str(row.get("direction", ""))
        sentence = "Supported this recommendation." if direction == "push_toward" else "Had weaker fit, but the model still selected this option."
        rows.append(f"""
        <div class='xai-shap-row'>
            <div class='xai-shap-icon'>↗</div>
            <div class='xai-shap-main'>
                <div class='xai-shap-topline'>
                    <span class='xai-shap-name'>{feature}</span>
                    <span class='xai-shap-value'>{imp:.3f}</span>
                </div>
                <div class='xai-shap-note'>{sentence}</div>
                <div class='xai-shap-track'><div class='xai-shap-fill' style='width:{width}%;'></div></div>
            </div>
        </div>
        """)

    _render_html(f"""
    <div class='xai-panel'>
        <div class='xai-panel-header'>
            <div>
                <div class='xai-panel-title'>📊 Feature Importance (SHAP)</div>
                <div class='xai-panel-subtitle'>These are the inputs that had the biggest impact on the model's recommendation.</div>
            </div>
            <span class='xai-pill-blue'>Global impact</span>
        </div>
        <div class='xai-shap-list'>
            {''.join(rows)}
        </div>
        <div class='xai-tip'>💡 <b>Tip:</b> Longer bars mean that factor had more influence on the recommendation.</div>
    </div>
    """)


def _render_path_walkthrough(ctx: dict, payload: dict):
    steps = _xai_path_steps(ctx, payload)
    recommended_name = payload.get("recommended_name", payload.get("recommended_id", "selected option"))
    cards = []
    for s in steps:
        if s["is_leaf"]:
            cards.append(f"""
            <div class='xai-path-card final'>
                <div class='xai-step-badge final'>Final result</div>
                <div class='xai-path-question'>Recommended option: {_xai_html(recommended_name)}</div>
                <div class='xai-path-text'>This is the end of the decision path for your inputs—the same final node as in the visual tree.</div>
            </div>
            """)
        else:
            cards.append(f"""
            <div class='xai-path-card selected'>
                <div class='xai-step-badge'>Step {s['step']}</div>
                <div class='xai-path-question'>{_xai_html(s['question'])}</div>
                <div class='xai-path-text'>{_xai_html(s['condition'])}</div>
                <div class='xai-path-branch'>Selected branch: <b>{_xai_html(s['branch'])}</b></div>
                <div class='xai-path-meta'>At this point, the tree's likely output is <b>{_xai_html(s['prediction'])}</b> based on {s['samples']} similar training cases.</div>
            </div>
            """)

    _render_html(f"""
    <div class='xai-panel'>
        <div class='xai-panel-header'>
            <div>
                <div class='xai-panel-title'>🧭 Step-by-step decision path</div>
                <div class='xai-panel-subtitle'>Same route as the visual decision tree: follow each split from the first rule to the final recommendation.</div>
            </div>
            <span class='xai-pill-green'>Selected path</span>
        </div>
        <div class='xai-path-list'>
            {''.join(cards)}
        </div>
    </div>
    """)


def _tree_layout_positions(tree_, node_id=0, depth=0, positions=None, leaf_counter=None):
    """Assign readable x/y positions to every node in the tree."""
    if positions is None:
        positions = {}
    if leaf_counter is None:
        leaf_counter = [0]

    left = int(tree_.children_left[node_id])
    right = int(tree_.children_right[node_id])
    if left == -1 and right == -1:
        x = leaf_counter[0]
        leaf_counter[0] += 1
        positions[node_id] = (x, -depth)
        return positions, leaf_counter

    positions, leaf_counter = _tree_layout_positions(tree_, left, depth + 1, positions, leaf_counter)
    positions, leaf_counter = _tree_layout_positions(tree_, right, depth + 1, positions, leaf_counter)
    x = (positions[left][0] + positions[right][0]) / 2.0
    positions[node_id] = (x, -depth)
    return positions, leaf_counter


def _shorten_label(text: str, max_len: int = 30) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


def _xai_display_prediction(ctx: dict, payload: dict, node_id: int) -> str:
    """Use the final app recommendation for selected-path nodes so the tree and result card never disagree."""
    recommended_name = payload.get("recommended_name", payload.get("recommended_id", "selected option"))
    path_set = set(int(n) for n in ctx.get("path_nodes", []))
    if int(node_id) in path_set:
        return str(recommended_name)
    return _xai_prediction_label(ctx, int(node_id))


def _wrap_tree_text(text: str, width: int = 18, max_lines: int = 3) -> str:
    """Wrap node text into short lines so boxes become taller, not wider."""
    import textwrap as _tw
    clean = str(text).replace("_", " ").strip()
    lines = _tw.wrap(clean, width=width, break_long_words=False, break_on_hyphens=False)
    if not lines:
        return ""
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip("…") + "…"
    return "\n".join(lines)


def plot_full_tree_with_selected_path(ctx: dict, payload: dict):
    """Draw the complete tree full-width and highlight the selected path in green.

    This version is optimized for readability: node text is wrapped into 4-5
    short lines, boxes are taller, vertical spacing is larger, and non-selected
    nodes use very short labels so they do not collide with each other.
    """
    tree_ = ctx["tree"]
    path_set = set(int(n) for n in ctx.get("path_nodes", []))
    leaf_id = int(ctx.get("leaf_id", -1))
    positions, leaf_counter = _tree_layout_positions(tree_, 0)
    n_leaves = max(1, leaf_counter[0])
    max_depth = max([-y for _, y in positions.values()] + [1])

    # Wider logical canvas + larger vertical gaps. It still renders full-width
    # in Streamlit, but the extra logical width prevents sibling boxes colliding.
    fig_w = max(22.0, min(30.0, n_leaves * 1.18))
    fig_h = max(11.0, max_depth * 2.45 + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=210)
    ax.axis("off")

    x_scale = 1.85 if n_leaves <= 16 else 1.62
    y_gap = 1.55

    def xy(node):
        x, y = positions[int(node)]
        return x * x_scale, y * y_gap

    # Edges first, under the nodes.
    for node_id in sorted(positions.keys()):
        x, y = xy(node_id)
        left = int(tree_.children_left[node_id])
        right = int(tree_.children_right[node_id])
        if left != -1:
            for child, branch_text in [
                (left, _xai_branch_label(ctx, node_id, left)),
                (right, _xai_branch_label(ctx, node_id, right)),
            ]:
                cx, cy = xy(child)
                selected_edge = node_id in path_set and child in path_set
                ax.plot(
                    [x, cx], [y - 0.34, cy + 0.46],
                    color="#16A34A" if selected_edge else "#DCE6F0",
                    linewidth=4.2 if selected_edge else 1.25,
                    zorder=1,
                    solid_capstyle="round",
                )
                mx, my = (x + cx) / 2.0, (y + cy) / 2.0
                ax.text(
                    mx, my + 0.10, _shorten_label(branch_text, 12),
                    ha="center", va="center", fontsize=8.8,
                    color="#166534" if selected_edge else "#64748B",
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor="#DCFCE7" if selected_edge else "#FFFFFF",
                        edgecolor="none",
                        alpha=0.96,
                    ),
                    zorder=2,
                )

    for node_id in sorted(positions.keys()):
        x, y = xy(node_id)
        details = _xai_split_details(ctx, node_id)
        stats = _xai_node_stats(ctx, node_id)
        in_path = int(node_id) in path_set
        is_leaf = int(node_id) == leaf_id
        display_pred = _xai_display_prediction(ctx, payload, int(node_id))

        if is_leaf:
            fc, ec, lw, title = "#DCFCE7", "#16A34A", 3.8, "FINAL"
        elif in_path:
            fc, ec, lw, title = "#ECFDF5", "#16A34A", 3.3, "SELECTED"
        else:
            fc, ec, lw, title = "#FFFFFF", "#CBD5E1", 1.2, "OTHER"

        if details["is_leaf"]:
            if in_path:
                label = (
                    f"{title}\n"
                    f"Recommendation:\n{_wrap_tree_text(display_pred, width=16, max_lines=2)}\n"
                    f"Cases: {stats['samples']}"
                )
                fs = 10.8
            else:
                # Keep non-path leaves short; they are context, not the main explanation.
                label = f"OTHER\n{_wrap_tree_text(display_pred, width=13, max_lines=2)}\nCases: {stats['samples']}"
                fs = 8.2
        else:
            if in_path:
                question = _wrap_tree_text(details["question"], width=22, max_lines=3)
                pred = _wrap_tree_text(display_pred, width=17, max_lines=2)
                label = f"{title}\n{question}\nLikely:\n{pred}\nCases: {stats['samples']}"
                fs = 10.4
            else:
                # Off-path internal boxes are intentionally compact to avoid clutter.
                question = _wrap_tree_text(details["question"], width=18, max_lines=2)
                pred = _wrap_tree_text(display_pred, width=13, max_lines=1)
                label = f"{title}\n{question}\nLikely: {pred}\nCases: {stats['samples']}"
                fs = 8.1

        ax.text(
            x, y, label,
            ha="center", va="center", fontsize=fs,
            color="#0F172A" if in_path else "#334155",
            linespacing=1.22,
            bbox=dict(
                boxstyle="round,pad=0.62,rounding_size=0.12",
                facecolor=fc,
                edgecolor=ec,
                linewidth=lw,
            ),
            zorder=3,
        )

    xmax = (n_leaves - 1) * x_scale
    ymin = -max_depth * y_gap - 0.95
    ax.set_xlim(-1.55, xmax + 1.55)
    ax.set_ylim(ymin, 1.05)
    fig.tight_layout(pad=0.2)
    return fig

def _svg_escape(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _svg_text_lines(
    text: str,
    width: int = 20,
    max_lines: int = 4,
    *,
    break_long_words: bool = False,
) -> list:
    import textwrap as _tw
    clean = str(text).replace("_", " ").strip()
    lines = _tw.wrap(
        clean,
        width=width,
        break_long_words=break_long_words,
        break_on_hyphens=False,
    )
    if not lines:
        return [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip("…") + "…"
    return lines



def _xai_format_threshold(feature: str, value: float) -> str:
    """Format numeric split values in a non-technical way."""
    fl = str(feature).lower()
    try:
        v = float(value)
    except Exception:
        return str(value)
    if "price" in fl or "budget" in fl:
        return f"${v:,.2f}"
    if "area" in fl or "size" in fl:
        return f"{v:,.0f}"
    if abs(v - round(v)) < 1e-9:
        return f"{v:,.0f}"
    return f"{v:,.2f}"


def _xai_clean_question(ctx: dict, node_id: int) -> str:
    """Readable full question used mainly in hover text."""
    details = _xai_split_details(ctx, int(node_id))
    if details.get("is_leaf"):
        return "Final recommendation"
    tree_ = ctx["tree"]
    feature_idx = int(tree_.feature[int(node_id)])
    feature = details.get("feature", "Feature")
    feature_names = ctx.get("feature_names", [])
    encoded_name = feature_names[feature_idx] if feature_idx < len(feature_names) else ""
    threshold = float(tree_.threshold[int(node_id)])
    feature_group_map = ctx.get("feature_group_map", {})
    for prefix, label in feature_group_map.items():
        if encoded_name.startswith(prefix + "_"):
            category = encoded_name[len(prefix) + 1:].replace("_", " ")
            return f"Is {label.lower()} '{category}'?"
    return f"Is {str(feature).lower()} ≤ {_xai_format_threshold(feature, threshold)}?"


def _xai_compact_question(ctx: dict, node_id: int) -> str:
    """Very short node label. Full explanation stays in the hover card."""
    details = _xai_split_details(ctx, int(node_id))
    if details.get("is_leaf"):
        return "Final output"

    tree_ = ctx["tree"]
    feature_idx = int(tree_.feature[int(node_id)])
    feature = str(details.get("feature", "Feature")).strip()
    feature_names = ctx.get("feature_names", [])
    encoded_name = feature_names[feature_idx] if feature_idx < len(feature_names) else ""
    threshold = float(tree_.threshold[int(node_id)])
    feature_group_map = ctx.get("feature_group_map", {})

    # One-hot split: show only the feature/category, not a long sentence.
    for prefix, label in feature_group_map.items():
        if encoded_name.startswith(prefix + "_"):
            category = encoded_name[len(prefix) + 1:].replace("_", " ")
            return f"{label}: {category}?"

    # Numeric split: compact and readable.
    return f"{feature} ≤ {_xai_format_threshold(feature, threshold)}?"


def _xai_user_value_text(ctx: dict, node_id: int) -> str:
    """Return the user's value for this node's feature, when available."""
    details = _xai_split_details(ctx, int(node_id))
    tree_ = ctx["tree"]
    feature_idx = int(tree_.feature[int(node_id)])
    if feature_idx < 0:
        return "Final recommendation"
    feature = details.get("feature", "Feature")
    feature_names = ctx.get("feature_names", [])
    encoded_name = feature_names[feature_idx] if feature_idx < len(feature_names) else ""
    x_dense = np.asarray(ctx.get("x_dense", []), dtype=float)
    raw_value = float(x_dense[0, feature_idx]) if x_dense.ndim == 2 and feature_idx < x_dense.shape[1] else None
    feature_group_map = ctx.get("feature_group_map", {})
    for prefix, label in feature_group_map.items():
        if encoded_name.startswith(prefix + "_"):
            category = encoded_name[len(prefix) + 1:].replace("_", " ")
            answer = "Yes" if raw_value is not None and raw_value > 0.5 else "No"
            return f"{label}: {answer} for '{category}'"
    if raw_value is None:
        return f"{feature}: not available"
    return f"{feature}: {_xai_format_threshold(feature, raw_value)}"


def _xai_selected_next_node(ctx: dict, node_id: int):
    path = [int(n) for n in ctx.get("path_nodes", [])]
    node_id = int(node_id)
    if node_id not in path:
        return None
    idx = path.index(node_id)
    return path[idx + 1] if idx + 1 < len(path) else None


def _xai_result_word_for_node(ctx: dict, node_id: int) -> str:
    """Return Yes/No for the selected branch at this node."""
    next_node = _xai_selected_next_node(ctx, int(node_id))
    if next_node is None:
        return "Final"
    label = str(_xai_branch_label(ctx, int(node_id), int(next_node))).strip().lower()
    if label.startswith("yes"):
        return "Yes"
    if label.startswith("no"):
        return "No"
    return "Selected"


def _xai_node_tooltip_text(ctx: dict, payload: dict, node_id: int) -> str:
    """Plain-language, fixed-structure tooltip/panel text for ordinary users."""
    node_id = int(node_id)
    details = _xai_split_details(ctx, node_id)
    stats = _xai_node_stats(ctx, node_id)
    path_set = set(int(n) for n in ctx.get("path_nodes", []))
    in_path = node_id in path_set
    is_leaf = node_id == int(ctx.get("leaf_id", -1)) or details.get("is_leaf")
    recommended_name = str(payload.get("recommended_name", payload.get("recommended_id", "selected option")))
    question = _xai_clean_question(ctx, node_id)
    user_value = _xai_user_value_text(ctx, node_id)
    result_word = _xai_result_word_for_node(ctx, node_id)

    if is_leaf and in_path:
        return (
            "Final recommendation\n"
            f"Recommended option:\n{recommended_name}\n"
            "Why it was selected:\nIt passed the previous rules on the green path.\n"
            f"Similar past cases:\n{stats['samples']} similar cases followed this path.\n"
            "Confidence note:\nThis recommendation is based on patterns in the training data, not a perfect rule."
        )

    if in_path:
        passed = "passed this rule" if result_word == "Yes" else "followed this branch"
        meaning = "the tree continued to the next rule on the green path."
        return (
            "Decision at this step\n"
            f"The system checked:\n{question}\n"
            f"User's answer:\n{user_value}\n"
            f"Result:\n{result_word}. This option {passed}.\n"
            f"What this means:\nBecause this answer matched the selected path, {meaning}"
        )

    return (
        "Not selected\n"
        "This option was not chosen because:\nIt did not follow the green decision path.\n"
        f"Rule at this point:\n{question}\n"
        "What happened:\nThe user's answers did not match this branch, so the tree ignored this option."
    )


def _interactive_tree_node_lines(ctx: dict, payload: dict, node_id: int) -> tuple[list, str]:
    """Return compact node text plus a fuller hover/click explanation.

    Keep the SVG box text intentionally short. The screenshot was unreadable
    because each node tried to show the rule, result, next path, and sample count.
    The box should be a summary; the hover card should carry the explanation.
    """
    node_id = int(node_id)
    details = _xai_split_details(ctx, node_id)
    stats = _xai_node_stats(ctx, node_id)
    path_set = set(int(n) for n in ctx.get("path_nodes", []))
    in_path = node_id in path_set
    is_leaf = node_id == int(ctx.get("leaf_id", -1)) or details.get("is_leaf")
    display_pred = str(_xai_display_prediction(ctx, payload, node_id)).replace("_", " ")
    feature = str(details.get("feature", "Decision")).strip()
    question = _xai_compact_question(ctx, node_id)

    if is_leaf:
        title = "Recommended" if in_path else "Other option"
        lines = [
            title,
            _shorten_label(display_pred, 23),
            f"n = {stats['samples']}",
        ]
        return lines, _xai_node_tooltip_text(ctx, payload, node_id)

    if in_path:
        lines = [
            _shorten_label(feature, 22),
            _shorten_label(question, 26),
            f"n = {stats['samples']}",
        ]
    else:
        lines = [
            "Other branch",
            _shorten_label(question, 26),
            f"n = {stats['samples']}",
        ]

    return lines, _xai_node_tooltip_text(ctx, payload, node_id)

def _build_interactive_svg_tree_html(ctx: dict, payload: dict) -> tuple[str, int]:
    """Build a stable SVG tree with hover preview. The original node does not scale, so there is no shaking/flicker."""
    tree_ = ctx["tree"]
    path_set = set(int(n) for n in ctx.get("path_nodes", []))
    leaf_id = int(ctx.get("leaf_id", -1))
    positions, leaf_counter = _tree_layout_positions(tree_, 0)
    n_leaves = max(1, leaf_counter[0])
    max_depth = max([-y for _, y in positions.values()] + [1])

    node_w = 178
    node_h = 106
    x_gap = 210 if n_leaves <= 16 else 185
    y_gap = 148
    margin_x = 90
    margin_top = 70
    margin_bottom = 80
    svg_w = int(max(1180, (n_leaves - 1) * x_gap + 2 * margin_x + node_w))
    svg_h = int(margin_top + max_depth * y_gap + margin_bottom + node_h)

    def xy(node):
        x, y = positions[int(node)]
        return margin_x + x * x_gap + node_w / 2, margin_top + (-y) * y_gap + node_h / 2

    edge_parts = []
    label_parts = []
    for node_id in sorted(positions.keys()):
        left = int(tree_.children_left[node_id])
        right = int(tree_.children_right[node_id])
        if left != -1:
            x0, y0 = xy(node_id)
            for child in (left, right):
                x1, y1 = xy(child)
                selected_edge = int(node_id) in path_set and int(child) in path_set
                cls = "edge selected" if selected_edge else "edge"
                edge_parts.append(
                    f"<line class='{cls}' x1='{x0:.1f}' y1='{y0 + node_h/2 - 8:.1f}' x2='{x1:.1f}' y2='{y1 - node_h/2 + 8:.1f}' />"
                )
                branch = _svg_escape(_shorten_label(_xai_branch_label(ctx, node_id, child), 16))
                mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                label_cls = "branch-label selected" if selected_edge else "branch-label"
                label_parts.append(
                    f"<text class='{label_cls}' x='{mx:.1f}' y='{my:.1f}' text-anchor='middle'>{branch}</text>"
                )

    node_parts = []
    for node_id in sorted(positions.keys()):
        x, y = xy(node_id)
        in_path = int(node_id) in path_set
        is_leaf = int(node_id) == leaf_id
        lines, hover_text = _interactive_tree_node_lines(ctx, payload, int(node_id))
        cls = "tree-node selected" if in_path else "tree-node other"
        if is_leaf:
            cls = "tree-node final"
        safe_hover = _svg_escape(hover_text)
        safe_label_one_line = _svg_escape(" | ".join(lines))
        rect_x = x - node_w / 2
        rect_y = y - node_h / 2
        text_y = rect_y + 22
        tspans = []
        for i, line in enumerate(lines[:6]):
            line_cls = "node-title" if i == 0 else "node-text"
            dy = 0 if i == 0 else 15
            tspans.append(
                f"<tspan class='{line_cls}' x='{x:.1f}' dy='{dy}'>{_svg_escape(line)}</tspan>"
            )
        node_parts.append(
            f"""
            <g class='{cls}' data-hover='{safe_hover}' aria-label='{safe_label_one_line}' tabindex='0'>
                <rect x='{rect_x:.1f}' y='{rect_y:.1f}' width='{node_w}' height='{node_h}' rx='10' ry='10'></rect>
                <text class='node-label' x='{x:.1f}' y='{text_y:.1f}' text-anchor='middle'>{''.join(tspans)}</text>
            </g>
            """
        )

    svg = f"""
    <svg class='xai-tree-svg' viewBox='0 0 {svg_w} {svg_h}' role='img' aria-label='Complete decision tree'>
        <g class='tree-edges'>{''.join(edge_parts)}</g>
        <g class='tree-branch-labels'>{''.join(label_parts)}</g>
        <g class='tree-nodes'>{''.join(node_parts)}</g>
    </svg>
    """

    html_doc = f"""
    <div class='xai-tree-shell'>
        <div class='tree-toolbar'>
            <button type='button' id='zoomIn'>＋ Zoom in</button>
            <button type='button' id='zoomOut'>－ Zoom out</button>
            <button type='button' id='zoomReset'>Reset</button>
            <span>Hover a box to read a larger version. Drag to pan after zooming.</span>
        </div>
        <div id='treeViewport' class='tree-viewport'>
            <div id='treeCanvas' class='tree-canvas'>
                {svg}
            </div>
            <div id='nodePreview' class='node-preview'>Hover over a box to magnify its text.</div>
        </div>
    </div>
    <style>
        .xai-tree-shell {{ width:100%; border:1px solid #E2E8F0; border-radius:16px; background:#FFFFFF; padding:12px; box-sizing:border-box; }}
        .tree-toolbar {{ display:flex; align-items:center; gap:8px; margin-bottom:10px; font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:#475569; font-size:13px; }}
        .tree-toolbar button {{ border:1px solid #CBD5E1; background:#F8FAFC; border-radius:10px; padding:7px 10px; font-weight:750; color:#0F172A; cursor:pointer; }}
        .tree-toolbar button:hover {{ background:#EEF2FF; border-color:#93C5FD; }}
        .tree-viewport {{ position:relative; width:100%; height:min(74vh, 780px); min-height:560px; overflow:hidden; background:#FFFFFF; border-radius:14px; }}
        .tree-canvas {{ transform-origin:50% 8%; transition:transform 140ms ease-out; cursor:grab; user-select:none; }}
        .tree-canvas.dragging {{ cursor:grabbing; transition:none; }}
        .xai-tree-svg {{ width:100%; height:auto; display:block; }}
        .edge {{ stroke:#D7E3F0; stroke-width:2.1; stroke-linecap:round; }}
        .edge.selected {{ stroke:#16A34A; stroke-width:6.0; }}
        .branch-label {{ font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size:15px; font-weight:700; fill:#64748B; }}
        .branch-label.selected {{ fill:#15803D; font-weight:900; }}
        .tree-node rect {{ fill:#FFFFFF; stroke:#CBD5E1; stroke-width:2.2; filter:drop-shadow(0 2px 3px rgba(15,23,42,0.07)); transition:stroke-width 120ms ease, filter 120ms ease, fill 120ms ease; }}
        .tree-node.selected rect {{ fill:#ECFDF5; stroke:#16A34A; stroke-width:4.0; }}
        .tree-node.final rect {{ fill:#DCFCE7; stroke:#16A34A; stroke-width:4.4; }}
        .tree-node:hover rect, .tree-node:focus rect {{ stroke:#2563EB; stroke-width:5.2; fill:#EFF6FF; filter:drop-shadow(0 8px 15px rgba(37,99,235,0.24)); }}
        .tree-node.selected:hover rect, .tree-node.final:hover rect, .tree-node.selected:focus rect, .tree-node.final:focus rect {{ stroke:#15803D; fill:#DCFCE7; filter:drop-shadow(0 8px 15px rgba(22,163,74,0.26)); }}
        .node-label {{ pointer-events:none; font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill:#0F172A; }}
        .node-title {{ font-size:15px; font-weight:950; letter-spacing:.02em; }}
        .node-text {{ font-size:15px; font-weight:760; }}
        .tree-node.other .node-title, .tree-node.other .node-text {{ fill:#334155; }}
        .node-preview {{ display:none; position:absolute; right:18px; top:18px; width:min(390px, 42vw); white-space:pre-line; z-index:20; background:#0F172A; color:#FFFFFF; border-radius:18px; padding:18px 20px; font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size:18px; line-height:1.45; font-weight:750; box-shadow:0 22px 45px rgba(15,23,42,0.30); pointer-events:none; }}
        .node-preview.show {{ display:block; }}
        @media (max-width:900px) {{ .tree-viewport {{ min-height:480px; height:62vh; }} .node-preview {{ width:calc(100% - 36px); left:18px; right:18px; font-size:16px; }} .tree-toolbar {{ flex-wrap:wrap; }} }}
    </style>
    <script>
    (function() {{
        const viewport = document.getElementById('treeViewport');
        const canvas = document.getElementById('treeCanvas');
        const preview = document.getElementById('nodePreview');
        const zoomIn = document.getElementById('zoomIn');
        const zoomOut = document.getElementById('zoomOut');
        const zoomReset = document.getElementById('zoomReset');
        let scale = 1, tx = 0, ty = 0;
        let dragging = false, startX = 0, startY = 0, baseX = 0, baseY = 0;
        function apply() {{ canvas.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(${{scale}})`; }}
        function setScale(next) {{ scale = Math.max(0.85, Math.min(2.4, next)); if (scale <= 1.01) {{ tx = 0; ty = 0; }} apply(); }}
        zoomIn.addEventListener('click', () => setScale(scale + 0.18));
        zoomOut.addEventListener('click', () => setScale(scale - 0.18));
        zoomReset.addEventListener('click', () => {{ scale = 1; tx = 0; ty = 0; apply(); }});
        viewport.addEventListener('wheel', (e) => {{ if (!e.ctrlKey && !e.metaKey) return; e.preventDefault(); setScale(scale + (e.deltaY < 0 ? 0.12 : -0.12)); }}, {{ passive:false }});
        viewport.addEventListener('mousedown', (e) => {{ if (scale <= 1.01) return; dragging = true; canvas.classList.add('dragging'); startX = e.clientX; startY = e.clientY; baseX = tx; baseY = ty; }});
        window.addEventListener('mousemove', (e) => {{ if (!dragging) return; tx = baseX + (e.clientX - startX); ty = baseY + (e.clientY - startY); apply(); }});
        window.addEventListener('mouseup', () => {{ dragging = false; canvas.classList.remove('dragging'); }});
        document.querySelectorAll('.tree-node').forEach(node => {{
            node.addEventListener('mouseenter', () => {{ preview.textContent = node.getAttribute('data-hover') || ''; preview.classList.add('show'); }});
            node.addEventListener('mouseleave', () => {{ preview.classList.remove('show'); }});
            node.addEventListener('focus', () => {{ preview.textContent = node.getAttribute('data-hover') || ''; preview.classList.add('show'); }});
            node.addEventListener('blur', () => {{ preview.classList.remove('show'); }});
        }});
        apply();
    }})();
    </script>
    """
    return html_doc, int(min(920, max(680, svg_h * 0.72 + 86)))


def _render_full_tree_panel(ctx: dict, payload: dict):
    import streamlit.components.v1 as components

    with st.container(border=True):
        st.markdown("### 🌳 Complete Decision Tree")
        st.markdown(
            """
            <div style='color:#475569; font-size:15px; line-height:1.55; margin-bottom:12px;'>
            This is the full tree used for the visual explanation. The <b style='color:#16A34A;'>green boxes and green lines</b>
            show the route followed from the first rule to the final recommendation above. Hover over any box to see a larger readable preview.
            </div>
            """,
            unsafe_allow_html=True,
        )
        tree_html, height = _build_interactive_svg_tree_html(ctx, payload)
        components.html(tree_html, height=height, scrolling=False)
        st.markdown(
            """
            <div style='border-left:4px solid #16A34A; background:#F0FDF4; padding:12px 14px; border-radius:12px; color:#14532D; font-size:14px; line-height:1.55; margin-top:12px;'>
            <b>How to read it:</b> Start at the top. At each question, follow the green line. The final green box is the same recommendation shown in the result card.
            </div>
            """,
            unsafe_allow_html=True,
        )

def _inject_xai_dashboard_css():
    st.markdown(
        """
        <style>
        .block-container { max-width: 1540px !important; }
        .xai-dashboard-title {
            font-size: 30px; font-weight: 850; color:#0F172A; margin-top: 18px; margin-bottom: 4px;
        }
        .xai-dashboard-subtitle {
            color:#334155; font-size:17px; font-weight:650; line-height:1.6; margin-bottom:18px;
        }
        .xai-panel {
            border:1px solid #DDE3EA; border-radius:18px; padding:20px 22px; background:#FFFFFF;
            box-shadow:0 10px 28px rgba(15,23,42,0.055); margin-bottom:18px;
        }
        .xai-panel-header { display:flex; align-items:flex-start; justify-content:space-between; gap:14px; margin-bottom:16px; }
        .xai-panel-title { font-size:20px; font-weight:850; color:#0F172A; margin-bottom:6px; }
        .xai-panel-subtitle { font-size:14px; color:#64748B; line-height:1.5; max-width:760px; }
        .xai-pill-blue { background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; border-radius:999px; padding:6px 10px; font-size:12px; font-weight:800; white-space:nowrap; }
        .xai-pill-green { background:#ECFDF5; color:#166534; border:1px solid #BBF7D0; border-radius:999px; padding:6px 10px; font-size:12px; font-weight:800; white-space:nowrap; }
        .xai-shap-row { display:flex; gap:13px; align-items:flex-start; padding:13px 0; border-bottom:1px solid #E5E7EB; }
        .xai-shap-row:last-child { border-bottom:none; }
        .xai-shap-icon { width:36px; height:36px; border-radius:12px; background:#EFF6FF; color:#2563EB; display:flex; align-items:center; justify-content:center; font-size:17px; font-weight:900; flex:0 0 36px; }
        .xai-shap-main { flex:1; min-width:0; }
        .xai-shap-topline { display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
        .xai-shap-name { font-size:15px; font-weight:800; color:#0F172A; }
        .xai-shap-value { font-size:14px; font-weight:850; color:#0F172A; }
        .xai-shap-note { font-size:12.5px; color:#64748B; margin-top:3px; }
        .xai-shap-track { height:8px; border-radius:999px; background:#E8EEF5; overflow:hidden; margin-top:8px; }
        .xai-shap-fill { height:8px; border-radius:999px; background:linear-gradient(90deg,#2563EB,#60A5FA); }
        .xai-tip { margin-top:14px; border:1px solid #E2E8F0; background:#F8FAFC; border-radius:14px; padding:12px 14px; color:#475569; font-size:13px; line-height:1.5; }
        .xai-path-list { position:relative; }
        .xai-path-card { border-radius:16px; padding:14px 16px; margin:12px 0; box-shadow:0 2px 10px rgba(15,23,42,0.045); }
        .xai-path-card.selected { border:2px solid #16A34A; background:#F0FDF4; }
        .xai-path-card.final { border:2px solid #16A34A; background:#DCFCE7; }
        .xai-step-badge { display:inline-block; color:#166534; background:#DCFCE7; border:1px solid #BBF7D0; border-radius:999px; padding:4px 9px; font-size:11px; font-weight:850; text-transform:uppercase; }
        .xai-step-badge.final { color:#14532D; background:#BBF7D0; }
        .xai-path-question { font-size:16px; font-weight:850; color:#0F172A; margin-top:8px; }
        .xai-path-text { font-size:13.5px; color:#475569; margin-top:6px; line-height:1.45; }
        .xai-path-branch { font-size:13.5px; color:#166534; margin-top:8px; }
        .xai-path-meta { font-size:12.5px; color:#64748B; margin-top:7px; line-height:1.45; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_visual_explanation(config: dict, payload: dict):
    """Final user-facing visual explanation: SHAP first, then readable full tree with selected path."""
    _inject_xai_dashboard_css()
    st.markdown("<div class='xai-dashboard-title'>How the model made this recommendation</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='xai-dashboard-subtitle'>These visualizations explain the main factors that influenced the model's prediction. SHAP shows feature influence; the decision tree shows the step-by-step route to the result.</div>",
        unsafe_allow_html=True,
    )

    ctx = _get_tree_path_context(payload, config)

    # SHAP first, then the complete decision tree full-width.
    # No separate step-by-step path card, because the tree itself is the explanation.
    _render_readable_shap_card(payload, max_items=8)
    if ctx is not None:
        _render_full_tree_panel(ctx, payload)
    else:
        st.warning("Decision tree could not be displayed. Re-train the model bundle with surrogate_tree included.")

# -----------------------------------------------------------------------------
# FINAL LAYOUT OVERRIDE: narrow centered result/SHAP + full-width CSS/SVG tree
# -----------------------------------------------------------------------------

def _inject_xai_dashboard_css():
    """Keep the main app centered; only the tree panel escapes to full browser width."""
    st.markdown(
        """
        <style>
        /* Do NOT widen Streamlit's main block container here. Forms/result/SHAP stay centered. */
        .xai-dashboard-title {
            font-size: 30px; font-weight: 850; color:#0F172A; margin-top: 18px; margin-bottom: 4px;
        }
        .xai-dashboard-subtitle {
            color:#334155; font-size:17px; font-weight:650; line-height:1.6; margin-bottom:18px;
        }
        .xai-panel {
            border:1px solid #DDE3EA; border-radius:18px; padding:20px 22px; background:#FFFFFF;
            box-shadow:0 10px 28px rgba(15,23,42,0.055); margin-bottom:18px;
        }
        .xai-panel-header { display:flex; align-items:flex-start; justify-content:space-between; gap:14px; margin-bottom:16px; }
        .xai-panel-title { font-size:20px; font-weight:850; color:#0F172A; margin-bottom:6px; }
        .xai-panel-subtitle { font-size:14px; color:#64748B; line-height:1.5; max-width:760px; }
        .xai-pill-blue { background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE; border-radius:999px; padding:6px 10px; font-size:12px; font-weight:800; white-space:nowrap; }
        .xai-shap-row { display:flex; gap:13px; align-items:flex-start; padding:13px 0; border-bottom:1px solid #E5E7EB; }
        .xai-shap-row:last-child { border-bottom:none; }
        .xai-shap-icon { width:36px; height:36px; border-radius:12px; background:#EFF6FF; color:#2563EB; display:flex; align-items:center; justify-content:center; font-size:17px; font-weight:900; flex:0 0 36px; }
        .xai-shap-main { flex:1; min-width:0; }
        .xai-shap-topline { display:flex; justify-content:space-between; gap:12px; align-items:baseline; }
        .xai-shap-name { font-size:15px; font-weight:800; color:#0F172A; }
        .xai-shap-value { font-size:14px; font-weight:850; color:#0F172A; }
        .xai-shap-note { font-size:12.5px; color:#64748B; margin-top:3px; }
        .xai-shap-track { height:8px; border-radius:999px; background:#E8EEF5; overflow:hidden; margin-top:8px; }
        .xai-shap-fill { height:8px; border-radius:999px; background:linear-gradient(90deg,#2563EB,#60A5FA); }
        .xai-tip { margin-top:14px; border:1px solid #E2E8F0; background:#F8FAFC; border-radius:14px; padding:12px 14px; color:#475569; font-size:13px; line-height:1.5; }
        .xai-pill-green { background:#ECFDF5; color:#166534; border:1px solid #BBF7D0; border-radius:999px; padding:6px 10px; font-size:12px; font-weight:800; white-space:nowrap; }
        .xai-path-list { position:relative; }
        .xai-path-card { border-radius:16px; padding:14px 16px; margin:12px 0; box-shadow:0 2px 10px rgba(15,23,42,0.045); }
        .xai-path-card.selected { border:2px solid #16A34A; background:#F0FDF4; }
        .xai-path-card.final { border:2px solid #16A34A; background:#DCFCE7; }
        .xai-step-badge { display:inline-block; color:#166534; background:#DCFCE7; border:1px solid #BBF7D0; border-radius:999px; padding:4px 9px; font-size:11px; font-weight:850; text-transform:uppercase; }
        .xai-step-badge.final { color:#14532D; background:#BBF7D0; }
        .xai-path-question { font-size:16px; font-weight:850; color:#0F172A; margin-top:8px; }
        .xai-path-text { font-size:13.5px; color:#475569; margin-top:6px; line-height:1.45; }
        .xai-path-branch { font-size:13.5px; color:#166534; margin-top:8px; }
        .xai-path-meta { font-size:12.5px; color:#64748B; margin-top:7px; line-height:1.45; }

        /* Only this section is full browser width. */
        .xai-tree-full-bleed {
            position: relative;
            left: 50%;
            right: 50%;
            width: 100vw;
            margin-left: -50vw;
            margin-right: -50vw;
            box-sizing: border-box;
            padding-left: 8px;
            padding-right: 8px;
            margin-top: 20px;
            margin-bottom: 18px;
        }

        .xai-continue-shell {
            max-width: 1560px;
            margin: 0 auto;
            padding: 0 22px 0 22px;
            box-sizing: border-box;
        }
        .xai-continue-shell [data-testid="stLinkButton"],
        .xai-continue-shell [data-testid="stButton"] {
            width: 100% !important;
            max-width: none !important;
        }
        .xai-continue-shell [data-testid="stLinkButton"] a,
        .xai-continue-shell [data-testid="stButton"] button,
        .xai-continue-shell a,
        .xai-continue-shell button {
            width: 100% !important;
            max-width: none !important;
            min-height: 54px !important;
            border-radius: 13px !important;
            font-size: 18px !important;
            font-weight: 850 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        .xai-tree-shell {
            width: 100%;
            max-width: none;
            margin: 0 auto;
            border: 1px solid #DDE3EA;
            border-radius: 18px;
            background: #FFFFFF;
            box-shadow: 0 10px 28px rgba(15,23,42,0.055);
            padding: 20px 22px 18px 22px;
        }
        .xai-tree-title { font-size:26px; font-weight:950; color:#0F172A; margin-bottom:8px; }
        .xai-tree-subtitle { color:#334155; font-size:16.5px; font-weight:650; line-height:1.6; margin-bottom:14px; }
        .xai-tree-canvas-wrap {
            border:1px solid #DDE7F2;
            border-radius:16px;
            background:#FFFFFF;
            overflow: visible;
            padding: 12px 10px;
        }
        /* Fit the entire tree inside the full-width tree section.
           No horizontal scrollbar: the SVG scales to the available browser width. */
        .xai-tree-svg {
            display:block;
            width:100%;
            max-width:100%;
            height:auto;
            overflow:visible;
        }
        .xai-tree-help {
            border-left:4px solid #16A34A;
            background:#F0FDF4;
            padding:12px 14px;
            border-radius:12px;
            color:#14532D;
            font-size:15.5px;
            font-weight:650;
            line-height:1.6;
            margin-top:12px;
        }
        /* Unitless font-size / stroke-width = SVG user units, so text and lines scale with the diagram
           (px values stayed screen-sized while boxes grew in user space, which looked like huge padding + tiny type). */
        .xai-tree-svg .edge { stroke:#94A3B8; stroke-width:4.2; stroke-linecap:round; }
        .xai-tree-svg .edge.selected { stroke:#16A34A; stroke-width:7.0; }
        .xai-tree-svg .branch-label { font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size:21; font-weight:950; fill:#475569; }
        .xai-tree-svg .branch-label.selected { fill:#166534; font-weight:950; }
        .xai-tree-svg .tree-node rect { transition:fill .16s ease, stroke .16s ease, filter .16s ease; }
        .xai-tree-svg .tree-node.other rect { fill:#FFFFFF; stroke:#94A3B8; stroke-width:2.8; }
        .xai-tree-svg .tree-node.selected rect { fill:#ECFDF5; stroke:#16A34A; stroke-width:4.6; }
        .xai-tree-svg .tree-node.final rect { fill:#DCFCE7; stroke:#16A34A; stroke-width:5.0; }
        .xai-tree-svg .tree-node:hover rect { fill:#EFF6FF; stroke:#2563EB; stroke-width:5.4; filter:drop-shadow(0 10px 18px rgba(37,99,235,0.20)); }
        .xai-tree-svg .tree-node.selected:hover rect, .xai-tree-svg .tree-node.final:hover rect { fill:#DCFCE7; stroke:#15803D; filter:drop-shadow(0 10px 18px rgba(22,163,74,0.24)); }
        .xai-tree-svg .node-title { font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size:26; font-weight:950; fill:#0F172A; pointer-events:none; }
        .xai-tree-svg .node-text { font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size:21; font-weight:800; fill:#0F172A; pointer-events:none; }
        .xai-tree-svg .tree-node.other .node-title, .xai-tree-svg .tree-node.other .node-text { fill:#334155; }
        .xai-tree-svg .hover-card { opacity:0; pointer-events:none; transition:opacity .14s ease; }
        .xai-tree-svg .tree-node:hover .hover-card, .xai-tree-svg .tree-node:focus .hover-card { opacity:1; }
        /* Tooltip panel: IDE-style dark card (cf. "New Branch"), scales with SVG user units */
        .xai-tree-svg .hover-box {
            fill:#3F3F46 !important;
            stroke:#52525B !important;
            stroke-width:2.2;
            filter:drop-shadow(0 12px 32px rgba(0,0,0,0.38));
        }
        .xai-tree-svg .hover-title {
            font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size:38;
            font-weight:950;
            fill:#FAFAFA !important;
            letter-spacing:0.01em;
        }
        .xai-tree-svg .hover-section {
            font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size:32;
            font-weight:850;
            fill:#F4F4F5 !important;
        }
        .xai-tree-svg .hover-body {
            font-family:Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size:31;
            font-weight:650;
            fill:#D4D4D8 !important;
        }
        @media (max-width: 900px) {
            .xai-tree-full-bleed { padding-left:10px; padding-right:10px; }
            .xai-tree-shell { padding:16px 14px; }
            .xai-tree-svg .node-title { font-size:29; }
            .xai-tree-svg .node-text { font-size:23; }
            .xai-tree-svg .hover-title { font-size:40; }
            .xai-tree-svg .hover-section { font-size:34; }
            .xai-tree-svg .hover-body { font-size:33; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _svg_multiline_text(
    lines,
    x,
    y,
    cls,
    line_gap=22,
    anchor="middle",
    font_size: Optional[float] = None,
) -> str:
    """Emit SVG <text> lines. Optional font_size sets a presentation attribute (user units) so sizing survives Streamlit/CSS quirks."""
    parts = []
    fs_attr = f' font-size="{font_size}"' if font_size is not None else ""
    for i, line in enumerate(lines):
        parts.append(
            f"<text class='{cls}'{fs_attr} x='{x:.1f}' y='{y + i * line_gap:.1f}' text-anchor='{anchor}'>{_svg_escape(line)}</text>"
        )
    return "".join(parts)



def _build_full_bleed_svg_tree_markup(ctx: dict, payload: dict) -> str:
    """Pure HTML/CSS/SVG tree. Hover/click preview is rendered above all nodes, never behind them."""
    tree_ = ctx["tree"]
    path_set = set(int(n) for n in ctx.get("path_nodes", []))
    leaf_id = int(ctx.get("leaf_id", -1))
    positions, leaf_counter = _tree_layout_positions(tree_, 0)
    n_leaves = max(1, leaf_counter[0])
    max_depth = max([-y for _, y in positions.values()] + [1])

    # Uniform node box tall enough for wrapped labels; long tokens break so text stays inside (clip backup).
    node_w = 272
    node_h = 138
    x_gap = 268 if n_leaves <= 16 else 248
    y_gap = 204
    margin_x = 52
    margin_top = 64
    margin_bottom = 80
    svg_w = int(max(1280, (n_leaves - 1) * x_gap + 2 * margin_x + node_w))
    svg_h = int(margin_top + max_depth * y_gap + margin_bottom + node_h)

    def xy(node):
        x, y = positions[int(node)]
        return margin_x + x * x_gap + node_w / 2, margin_top + (-y) * y_gap + node_h / 2

    edge_parts = []
    label_parts = []
    for node_id in sorted(positions.keys()):
        left = int(tree_.children_left[node_id])
        right = int(tree_.children_right[node_id])
        if left != -1:
            x0, y0 = xy(node_id)
            for child in (left, right):
                x1, y1 = xy(child)
                selected_edge = int(node_id) in path_set and int(child) in path_set
                cls = "edge selected" if selected_edge else "edge"
                edge_parts.append(
                    f"<line class='{cls}' x1='{x0:.1f}' y1='{y0 + node_h/2 - 8:.1f}' x2='{x1:.1f}' y2='{y1 - node_h/2 + 8:.1f}' />"
                )
                branch = _shorten_label(_xai_branch_label(ctx, node_id, child), 18)
                mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                label_cls = "branch-label selected" if selected_edge else "branch-label"
                label_parts.append(
                    f"<text class='{label_cls}' font-size='21' x='{mx:.1f}' y='{my:.1f}' text-anchor='middle'>{_svg_escape(branch)}</text>"
                )

    node_parts = []
    node_clip_defs = []
    hover_parts = []
    hover_clip_defs = []
    hover_css_rules = []
    for node_id in sorted(positions.keys()):
        x, y = xy(node_id)
        in_path = int(node_id) in path_set
        is_leaf = int(node_id) == leaf_id
        lines, hover_text = _interactive_tree_node_lines(ctx, payload, int(node_id))
        lines = lines[:3]
        node_cls = f"tree-node node-{int(node_id)} selected" if in_path else f"tree-node node-{int(node_id)} other"
        if is_leaf:
            node_cls = f"tree-node node-{int(node_id)} final"

        rect_x = x - node_w / 2
        rect_y = y - node_h / 2
        node_fs_title = 26.0
        node_fs_body = 21.0
        # Narrow wrap + break long tokens so labels never spill past the cube horizontally.
        title_lines = _svg_text_lines(
            lines[0] if lines else "", width=16, max_lines=3, break_long_words=True
        )
        body_lines: list[str] = []
        for line in lines[1:]:
            body_lines.extend(
                _svg_text_lines(line, width=18, max_lines=4, break_long_words=True)
            )
        body_lines = body_lines[:6]

        total_lines = len(title_lines) + len(body_lines)
        line_gap = 21.0
        # Slight upward nudge: baseline-centered block was sitting low and clipped the last line.
        text_start_y = y - ((total_lines - 1) * line_gap) / 2.0 - (0.32 * node_fs_body)
        node_text = _svg_multiline_text(
            title_lines,
            x,
            text_start_y,
            "node-title",
            line_gap=line_gap,
            font_size=node_fs_title,
        )
        node_text += _svg_multiline_text(
            body_lines,
            x,
            text_start_y + len(title_lines) * line_gap,
            "node-text",
            line_gap=line_gap,
            font_size=node_fs_body,
        )

        node_clip_defs.append(
            f'<clipPath id="nclip-{int(node_id)}">'
            f'<rect x="{rect_x:.1f}" y="{rect_y:.1f}" width="{node_w}" height="{node_h}" rx="11" ry="11"/>'
            f"</clipPath>"
        )

        node_parts.append(f"""
        <g class='{node_cls}' tabindex='0'>
            <rect x='{rect_x:.1f}' y='{rect_y:.1f}' width='{node_w}' height='{node_h}' rx='11' ry='11'></rect>
            <g clip-path="url(#nclip-{int(node_id)})">
            {node_text}
            </g>
        </g>
        """)

        # Hover: no blank "separator" lines (skip empty paragraphs), tight wrap so lines stay inside the card,
        # clip-path on text so nothing paints outside the rounded rect. Font sizes +2 vs previous.
        hover_wrap = 25
        hover_lines: list[str] = []
        for block in str(hover_text).split("\n"):
            b = block.strip()
            if not b:
                continue
            hover_lines.extend(_svg_text_lines(b, width=hover_wrap, max_lines=14))
        hover_lines = hover_lines[:32]

        hover_section_headers = frozenset({
            "decision at this step", "final recommendation", "not selected", "the system checked",
            "user's answer", "result", "what this means", "recommended option", "why it was selected",
            "similar past cases", "confidence note", "rule at this point", "what happened",
            "this option was not chosen because",
        })

        # Larger type + line leading ≈ font so the card fills vertically; modest bottom pad to avoid a big empty band.
        hover_pad_top = 46
        hover_pad_bottom = 16
        hover_after_title = 42
        hover_after_section = 36
        hover_after_body = 35
        hover_fs_title = 38.0
        hover_fs_section = 32.0
        hover_fs_body = 31.0

        def _hover_panel_height(lines: list) -> int:
            yy = hover_pad_top
            seen_title = False
            for line in lines:
                if not line.strip():
                    continue
                lower = line.lower().strip().rstrip(":")
                if not seen_title:
                    seen_title = True
                    yy += hover_after_title
                elif lower in hover_section_headers:
                    yy += hover_after_section
                else:
                    yy += hover_after_body
            return int(yy + hover_pad_bottom)

        hover_w = 600
        hover_h = _hover_panel_height(hover_lines)
        # Prefer showing the card above/right of the node, but keep it inside the SVG viewBox.
        hx = x + 28
        if hx + hover_w > svg_w - 16:
            hx = x - hover_w - 28
        hx = min(max(hx, 16), svg_w - hover_w - 16)
        hy = rect_y - hover_h - 24
        if hy < 16:
            hy = rect_y + node_h + 24
        hy = min(max(hy, 16), svg_h - hover_h - 16)

        pad_x = 18
        clip_inset = 5.0
        hover_clip_defs.append(
            f'<clipPath id="hhover-{int(node_id)}">'
            f'<rect x="{hx + clip_inset:.1f}" y="{hy + clip_inset:.1f}" '
            f'width="{hover_w - 2 * clip_inset:.1f}" height="{hover_h - 2 * clip_inset:.1f}" rx="9" ry="9"/>'
            f"</clipPath>"
        )

        text_parts = []
        yy = hy + hover_pad_top
        seen_title = False
        for line in hover_lines:
            if not line.strip():
                continue
            lower = line.lower().strip().rstrip(":")
            if not seen_title:
                cls = "hover-title"
                fs = hover_fs_title
                seen_title = True
                yy_step = hover_after_title
            elif lower in hover_section_headers:
                cls = "hover-section"
                fs = hover_fs_section
                yy_step = hover_after_section
            else:
                cls = "hover-body"
                fs = hover_fs_body
                yy_step = hover_after_body
            text_parts.append(
                f"<text class='{cls}' font-size='{fs}' x='{hx + pad_x:.1f}' y='{yy:.1f}' text-anchor='start'>{_svg_escape(line)}</text>"
            )
            yy += yy_step

        hover_parts.append(f"""
        <g class='hover-card hover-for-{int(node_id)}'>
            <rect class='hover-box' x='{hx:.1f}' y='{hy:.1f}' width='{hover_w}' height='{hover_h}' rx='12' ry='12'></rect>
            <g clip-path="url(#hhover-{int(node_id)})">
            {''.join(text_parts)}
            </g>
        </g>
        """)
        hover_css_rules.append(
            f".xai-tree-svg:has(.node-{int(node_id)}:hover) .hover-for-{int(node_id)}, "
            f".xai-tree-svg:has(.node-{int(node_id)}:focus) .hover-for-{int(node_id)} {{ opacity:1; }}"
        )

    dynamic_hover_css = "\n".join(hover_css_rules)

    return f"""
    <div class='xai-tree-full-bleed'>
        <div class='xai-tree-shell'>
            <div class='xai-tree-title'>🌳 Complete Decision Tree</div>
            <div class='xai-tree-subtitle'>
                This is the full tree used for the visual explanation. The <b style='color:#16A34A;'>green boxes and green lines</b>
                show the route followed from the first rule to the final recommendation above. Hover or click any box to see a larger readable explanation.
            </div>
            <div class='xai-tree-canvas-wrap'>
                <svg class='xai-tree-svg' width='{svg_w}' height='{svg_h}' viewBox='0 0 {svg_w} {svg_h}' role='img' aria-label='Complete decision tree with selected path highlighted'>
                    <defs>{''.join(node_clip_defs)}{''.join(hover_clip_defs)}</defs>
                    <g class='edge-layer'>{''.join(edge_parts)}</g>
                    <g class='branch-layer'>{''.join(label_parts)}</g>
                    <g class='node-layer'>{''.join(node_parts)}</g>
                    <g class='hover-layer'>{''.join(hover_parts)}</g>
                </svg>
            </div>
            <div class='xai-tree-help'>
                <b>How to read it:</b> Start at the top. At each question, follow the green line. The final green box is the same recommendation shown in the result card.
            </div>
        </div>
    </div>
    <style>{dynamic_hover_css}</style>
    """

def _render_full_tree_panel(ctx: dict, payload: dict):
    # Render as normal HTML/SVG, not a Streamlit component iframe. This lets only the tree section become full-width.
    _render_html(_build_full_bleed_svg_tree_markup(ctx, payload))


def _render_visual_explanation(config: dict, payload: dict):
    """Final visual explanation: decision tree first, then SHAP summary."""
    _inject_xai_dashboard_css()
    st.markdown("<div class='xai-dashboard-title'>How the model made this recommendation</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='xai-dashboard-subtitle'>First, review the decision tree showing how the recommendation was generated. Then, examine the SHAP explanation graph highlighting the most influential input factors. You will answer questions related to these explanations in the next section.</div>",
        unsafe_allow_html=True,
    )

    ctx = _get_tree_path_context(payload, config)

    # 1) Decision tree first
    if ctx is not None:
        _render_full_tree_panel(ctx, payload)
    else:
        st.warning("Decision tree could not be displayed. Re-train the model bundle with surrogate_tree included.")

    # 2) SHAP summary after the tree
    _render_readable_shap_card(payload, max_items=8)


def render_full_width_continue_button(
    route: dict,
    survey_map: dict,
    payload: dict,
    task_name: str,
    label: str = "Continue to Survey",
):
    """Render the survey button as a full-browser-width section, matching the decision-tree width."""
    _inject_xai_dashboard_css()
    st.markdown("<div class='xai-tree-full-bleed'><div class='xai-continue-shell'>", unsafe_allow_html=True)
    render_timed_continue_to_survey(route, survey_map, payload, task_name, label=label)
    st.markdown("</div></div>", unsafe_allow_html=True)
