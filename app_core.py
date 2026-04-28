import re
import time
from pathlib import Path
from urllib.parse import urlencode

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn import tree


VALID_GROUPS = {"visual", "text"}
VALID_APPS = {"app_a", "app_b", "app_c"}
VALID_STEPS = {"1", "2", "3"}


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
            div[data-testid="stDialog"] > div {
                border-radius: 16px !important;
                border: 1px solid #E5E7EB !important;
                background: #FFFFFF !important;
                color: #111827 !important;
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
                min-height: 118px;
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
        dialog_decorator = st.dialog("Welcome")
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
    fig = plt.figure(figsize=(18.5, fig_height), dpi=150)
    gs = fig.add_gridspec(
        nrows=1,
        ncols=3,
        width_ratios=[1.08, 1.22, 0.06],
        wspace=0.38,
    )
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_swarm = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

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
    ax_swarm.tick_params(axis="y", length=0)
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

    fig.subplots_adjust(left=0.13, right=0.975, top=0.88, bottom=0.16)
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
        "<div class='mm-section-title'>Before seeing the AI explanation</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='mm-section-subtitle'>Please rate the importance of each feature in the AI’s decision on a scale from 1 (not important at all) to 7 (significantly important).</div>",
        unsafe_allow_html=True,
    )

    ratings = {}
    all_answered = True

    for feature in feature_labels:
        safe_feature = (
            feature.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )
        key = f"{state_key}_{safe_feature}"

        # Question + 1–7 scale on one row (equal halves: room for full question + scale).
        outer = st.columns([6, 6], gap="small")

        with outer[0]:
            st.markdown(
                f"<div class='mm-feature-label'>{feature} was important in the AI’s decision.</div>",
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

    all_features = _top_features(payload)
    if all_features:
        with st.expander("See all factors ranked by TreeSHAP importance", expanded=False):
            for i, feature in enumerate(all_features, start=1):
                st.markdown(f"{i}. **{feature}**")

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

def _render_text_explanation(config: dict, payload: dict):
    st.subheader("Why this recommendation was made")
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