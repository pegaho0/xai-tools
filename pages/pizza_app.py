import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

import shap
from streamlit_sortables import sort_items

st.set_page_config(page_title="Pizza Recommendation Study", layout="centered")

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

def get_state_value(name: str, default: str = "") -> str:
    value = st.session_state.get(name, default)
    if value is None:
        return default
    return str(value).strip()

PID = get_state_value("pid")
GROUP = get_state_value("group")
APP1 = get_state_value("app1")
APP2 = get_state_value("app2")
APP3 = get_state_value("app3")
STEP = get_state_value("step")
APP_NAME = get_state_value("app")

POST_SURVEY_MAP = {
    "1": "https://concordia.yul1.qualtrics.com/jfe/form/SV_2cps8pBYmqBoJxk",
    "2": "https://concordia.yul1.qualtrics.com/jfe/form/SV_SURVEY2_ID",
    "3": "https://concordia.yul1.qualtrics.com/jfe/form/SV_SURVEY3_ID",
}

VALID_GROUPS = {"visual", "text"}
VALID_APPS = {"app_a", "app_b", "app_c"}
VALID_STEPS = {"1", "2", "3"}

errors = []
if not PID:
    errors.append("Missing pid")
if GROUP not in VALID_GROUPS:
    errors.append("Invalid group")
if APP1 not in VALID_APPS:
    errors.append("Invalid app1")
if APP2 not in VALID_APPS:
    errors.append("Invalid app2")
if APP3 not in VALID_APPS:
    errors.append("Invalid app3")
if STEP not in VALID_STEPS:
    errors.append("Invalid step")
if APP_NAME not in VALID_APPS:
    errors.append("Invalid app")

expected_app = {"1": APP1, "2": APP2, "3": APP3}.get(STEP)
if expected_app and APP_NAME != expected_app:
    errors.append(f"Expected app {expected_app} for step {STEP}, got {APP_NAME}")

if errors:
    st.error("Missing routing data. Please start from the main study link or Qualtrics.")
    st.stop()

PIZZAS = [
    {"pizza_id": "MARGHERITA", "name": "Margherita", "style": "Italian", "ingredient": "Cheese", "dietary_tag": "Vegetarian", "price": 18, "customer_rating": 4.3, "free_delivery": "Yes"},
    {"pizza_id": "PEPPERONI", "name": "Pepperoni", "style": "American", "ingredient": "Pepperoni", "dietary_tag": "None", "price": 22, "customer_rating": 4.6, "free_delivery": "No"},
    {"pizza_id": "VEGGIE", "name": "Veggie Supreme", "style": "American", "ingredient": "Vegetables", "dietary_tag": "Vegetarian", "price": 20, "customer_rating": 4.4, "free_delivery": "Yes"},
    {"pizza_id": "CHICKEN_DELUXE", "name": "Chicken Deluxe", "style": "American", "ingredient": "Chicken", "dietary_tag": "None", "price": 26, "customer_rating": 4.7, "free_delivery": "No"},
    {"pizza_id": "MUSHROOM_TRUFFLE", "name": "Mushroom Truffle", "style": "Italian", "ingredient": "Mushrooms", "dietary_tag": "Vegetarian", "price": 28, "customer_rating": 4.8, "free_delivery": "Yes"},
    {"pizza_id": "GLUTEN_FREE_GARDEN", "name": "Gluten-Free Garden", "style": "Italian", "ingredient": "Vegetables", "dietary_tag": "Gluten-free", "price": 31, "customer_rating": 4.5, "free_delivery": "No"},
    {"pizza_id": "VEGAN_CLASSIC", "name": "Vegan Classic", "style": "Italian", "ingredient": "Vegetables", "dietary_tag": "Vegan", "price": 24, "customer_rating": 4.2, "free_delivery": "Yes"},
    {"pizza_id": "DAIRY_FREE_CHICKEN", "name": "Dairy-Free Chicken", "style": "American", "ingredient": "Chicken", "dietary_tag": "Dairy-free", "price": 29, "customer_rating": 4.4, "free_delivery": "Yes"},
]
PIZZA_ID_TO_META = {p["pizza_id"]: p for p in PIZZAS}

CAT_FEATURES = [
    "pizza_style",
    "ingredient_preference",
    "dietary_restriction_model",
    "rating_importance",
    "free_delivery_importance",
]
NUM_FEATURES = ["max_price"]

MENTAL_MODEL_FEATURES = [
    "Maximum price",
    "Pizza style",
    "Ingredient preference",
    "Dietary restriction / allergy",
    "Importance of customer rating",
    "Importance of free delivery",
]

FEATURE_GROUP_MAP = {
    "max_price": "Maximum price",
    "pizza_style": "Pizza style",
    "ingredient_preference": "Ingredient preference",
    "dietary_restriction_model": "Dietary restriction / allergy",
    "rating_importance": "Importance of customer rating",
    "free_delivery_importance": "Importance of free delivery",
}

RATING_IMPORTANCE_TO_WEIGHT = {
    "Not important": 0.0,
    "Slightly important": 0.7,
    "Moderately important": 1.4,
    "Very important": 2.1,
    "Extremely important": 2.8,
}
FREE_DELIVERY_IMPORTANCE_TO_WEIGHT = {
    "Not important": 0.0,
    "Slightly important": 0.8,
    "Moderately important": 1.5,
    "Very important": 2.2,
    "Extremely important": 3.0,
}
DIETARY_OPTIONS = [
    "None", "Vegetarian", "Vegan", "Gluten-free", "Dairy-free", "Other (please specify)"
]

def normalize_dietary_for_model(user_choice: str) -> str:
    return "Other" if user_choice == "Other (please specify)" else user_choice

def is_compatible(pizza_tag: str, user_restriction: str) -> bool:
    if user_restriction in ["None", "Other"]:
        return True
    if user_restriction == "Vegetarian":
        return pizza_tag in ["Vegetarian", "Vegan"]
    if user_restriction == "Vegan":
        return pizza_tag == "Vegan"
    if user_restriction == "Gluten-free":
        return pizza_tag == "Gluten-free"
    if user_restriction == "Dairy-free":
        return pizza_tag in ["Dairy-free", "Vegan"]
    return True

def choose_budget_safe_fallback(user_inputs: dict) -> dict:
    candidates = [p for p in PIZZAS if p["price"] <= user_inputs["max_price"]]
    if not candidates:
        candidates = [min(PIZZAS, key=lambda x: x["price"])]

    compatible = [p for p in candidates if is_compatible(p["dietary_tag"], user_inputs["dietary_restriction_model"])]
    if compatible:
        candidates = compatible

    def sim(p):
        s = 0.0
        s += 1.4 if p["style"] == user_inputs["pizza_style"] else 0.0
        s += 1.8 if p["ingredient"] == user_inputs["ingredient_preference"] else 0.0
        s += 2.4 if is_compatible(p["dietary_tag"], user_inputs["dietary_restriction_model"]) else -2.4
        s += RATING_IMPORTANCE_TO_WEIGHT[user_inputs["rating_importance"]] * (p["customer_rating"] - 4.0)
        s += FREE_DELIVERY_IMPORTANCE_TO_WEIGHT[user_inputs["free_delivery_importance"]] if p["free_delivery"] == "Yes" else 0.0
        s -= 0.05 * p["price"]
        return s

    return max(candidates, key=sim)

def _to_dense_1d(mat):
    if hasattr(mat, "toarray"):
        return np.asarray(mat.toarray()).ravel()
    return np.asarray(mat).ravel()

def base_feature_from_encoded_name(name: str) -> str:
    for prefix, label in FEATURE_GROUP_MAP.items():
        if name == prefix or name.startswith(prefix + "_"):
            return label
    return name

def aggregate_shap_to_study_features(shap_df: pd.DataFrame) -> pd.DataFrame:
    temp = shap_df.copy()
    temp["study_feature"] = temp["feature"].apply(base_feature_from_encoded_name)
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

def ranking_list_to_rank_dict(items):
    return {feature: idx + 1 for idx, feature in enumerate(items)}

def generate_synthetic_training_data(n=3500, seed=7):
    rng = np.random.default_rng(seed)
    max_price = rng.integers(15, 51, size=n)
    pizza_style = rng.choice(["Italian", "American"], size=n, p=[0.55, 0.45])
    ingredient_preference = rng.choice(
        ["Cheese", "Pepperoni", "Chicken", "Vegetables", "Mushrooms"],
        size=n, p=[0.18, 0.22, 0.20, 0.25, 0.15]
    )
    dietary_restriction_model = rng.choice(
        ["None", "Vegetarian", "Vegan", "Gluten-free", "Dairy-free", "Other"],
        size=n, p=[0.38, 0.20, 0.10, 0.10, 0.10, 0.12]
    )
    rating_importance = rng.choice(list(RATING_IMPORTANCE_TO_WEIGHT.keys()), size=n, p=[0.08, 0.15, 0.30, 0.27, 0.20])
    free_delivery_importance = rng.choice(list(FREE_DELIVERY_IMPORTANCE_TO_WEIGHT.keys()), size=n, p=[0.08, 0.14, 0.28, 0.28, 0.22])

    labels = []
    for i in range(n):
        mp = int(max_price[i])
        stl = pizza_style[i]
        ing = ingredient_preference[i]
        dr = dietary_restriction_model[i]
        ri = rating_importance[i]
        fdi = free_delivery_importance[i]

        affordable = [p for p in PIZZAS if p["price"] <= mp]
        if not affordable:
            affordable = [min(PIZZAS, key=lambda x: x["price"])]

        compatible = [p for p in affordable if is_compatible(p["dietary_tag"], dr)]
        candidates = compatible if compatible else affordable

        def score(p):
            s = 0.0
            s += 2.2 if p["style"] == stl else 0.0
            s += 2.6 if p["ingredient"] == ing else 0.0
            s += 3.5 if is_compatible(p["dietary_tag"], dr) else -3.5
            s += RATING_IMPORTANCE_TO_WEIGHT[ri] * (p["customer_rating"] - 4.0) * 2.2
            s += FREE_DELIVERY_IMPORTANCE_TO_WEIGHT[fdi] if p["free_delivery"] == "Yes" else 0.0
            gap = max(0, p["price"] - mp)
            s -= 1.8 * gap
            s -= 0.05 * p["price"]
            s += rng.normal(0, 0.35)
            return s

        best = max(candidates, key=score)
        labels.append(best["pizza_id"])

    X = pd.DataFrame({
        "max_price": max_price,
        "pizza_style": pizza_style,
        "ingredient_preference": ingredient_preference,
        "dietary_restriction_model": dietary_restriction_model,
        "rating_importance": rating_importance,
        "free_delivery_importance": free_delivery_importance,
    })
    y = pd.Series(labels, name="pizza_id")
    return X, y

@st.cache_resource
def train_model_and_explainer():
    X, y = generate_synthetic_training_data()

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    clf = LogisticRegression(max_iter=4000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    feature_names = pipe.named_steps["pre"].get_feature_names_out().tolist()
    feature_names = [n.replace("cat__", "").replace("num__", "") for n in feature_names]

    X_bg = X.sample(n=min(300, len(X)), random_state=42)
    X_bg_trans = pipe.named_steps["pre"].transform(X_bg)

    explainer = shap.LinearExplainer(
        pipe.named_steps["clf"],
        X_bg_trans,
        feature_perturbation="interventional",
    )
    return pipe, explainer, feature_names

model, explainer, FEATURE_NAMES = train_model_and_explainer()

def compute_shap_for_row(pipe: Pipeline, explainer: shap.LinearExplainer, x_row: pd.DataFrame):
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    X_trans = pre.transform(x_row)
    x_vec = _to_dense_1d(X_trans)

    pred_class = pipe.predict(x_row)[0]
    class_idx = list(clf.classes_).index(pred_class)

    shap_values = explainer.shap_values(X_trans)
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

    n = min(len(FEATURE_NAMES), len(x_vec), len(sv))
    df = pd.DataFrame(
        {"feature": FEATURE_NAMES[:n], "value": x_vec[:n], "shap_value": sv[:n]}
    )
    is_numeric = df["feature"].isin(NUM_FEATURES)
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

if "result_ready" not in st.session_state:
    st.session_state.result_ready = False
if "result_payload" not in st.session_state:
    st.session_state.result_payload = None
if "mental_model_order" not in st.session_state:
    st.session_state.mental_model_order = MENTAL_MODEL_FEATURES.copy()

SORTABLE_STYLE = """
.sortable-component { background-color: transparent; padding: 0; border: none; }
.sortable-container { background-color: transparent; padding: 0; border: none; counter-reset: rankitem; }
.sortable-container-header { display: none; }
.sortable-container-body { background-color: transparent; padding: 0; }
.sortable-item {
    position: relative;
    display: flex;
    align-items: center;
    min-height: 58px;
    margin: 0 0 12px 26px;
    padding: 10px 14px 10px 42px;
    background: #fafafa;
    border: 1px solid #d5d9de;
    border-radius: 8px;
    font-size: 16px;
    font-weight: 600;
    color: #1f2937;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.sortable-item::before {
    counter-increment: rankitem;
    content: counter(rankitem) ".";
    position: absolute;
    left: -22px;
    top: 50%;
    transform: translateY(-50%);
    color: #111827;
    font-size: 17px;
    font-weight: 700;
}
.sortable-item::after {
    content: "⋮⋮";
    position: absolute;
    left: 14px;
    top: 50%;
    transform: translateY(-50%);
    color: #9aa0a6;
    font-size: 18px;
    letter-spacing: -2px;
}
"""

GROUP_LABEL_MAP = {
    "visual": "Visual Explanation",
    "text": "Text Explanation",
}
st.title("🍕 Pizza Recommendation")
CONDITION_LABEL = GROUP_LABEL_MAP.get(GROUP, "Unknown Condition")
st.caption(f"{CONDITION_LABEL} • Step {STEP} of 3")
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
    list(RATING_IMPORTANCE_TO_WEIGHT.keys()),
    index=2,
)
free_delivery_importance = st.selectbox(
    "How important is free delivery when choosing a pizza?",
    list(FREE_DELIVERY_IMPORTANCE_TO_WEIGHT.keys()),
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

st.subheader("Before seeing the AI explanation")
st.caption("Drag the items to rank them from most influential to least influential in the AI recommendation.")
st.caption("Top = most influential • Bottom = least influential")

sorted_items = sort_items(
    [{"header": "", "items": st.session_state.mental_model_order}],
    multi_containers=True,
    custom_style=SORTABLE_STYLE,
    key="mental_model_sortable",
)

if isinstance(sorted_items, list) and len(sorted_items) > 0 and isinstance(sorted_items[0], dict):
    if "items" in sorted_items[0]:
        st.session_state.mental_model_order = sorted_items[0]["items"]

mental_model_ranks = ranking_list_to_rank_dict(st.session_state.mental_model_order)

if st.button("Get recommendation", type="primary", use_container_width=True):
    pred_id = model.predict(x)[0]
    meta = PIZZA_ID_TO_META[pred_id]

    if (meta["price"] > max_price) or (not is_compatible(meta["dietary_tag"], dietary_restriction_model)):
        meta = choose_budget_safe_fallback(x.iloc[0].to_dict())
        pred_id = meta["pizza_id"]

    _, base_value, shap_df = compute_shap_for_row(model, explainer, x)
    xai_agg = aggregate_shap_to_study_features(shap_df)
    xai_rank_list = xai_agg["study_feature"].tolist()

    st.session_state.result_ready = True
    st.session_state.result_payload = {
        "timestamp": int(time.time()),
        "participant_id": PID,
        "condition": GROUP,
        "inputs": {
            "max_price": max_price,
            "pizza_style": pizza_style,
            "ingredient_preference": ingredient_preference,
            "dietary_restriction": dietary_restriction,
            "dietary_restriction_other_text": dietary_restriction_other_text,
            "dietary_restriction_model": dietary_restriction_model,
            "rating_importance": rating_importance,
            "free_delivery_importance": free_delivery_importance,
        },
        "mental_model_order": st.session_state.mental_model_order.copy(),
        "mental_model_ranks": mental_model_ranks,
        "recommended_pizza_id": pred_id,
        "recommended_pizza_name": meta["name"],
        "recommended_pizza_price": meta["price"],
        "meta": meta,
        "shap_df": shap_df,
        "base_value": base_value,
        "xai_agg": xai_agg,
        "xai_rank_list": xai_rank_list,
    }
    st.rerun()

if st.session_state.result_ready and st.session_state.result_payload is not None:
    payload = st.session_state.result_payload
    meta = payload["meta"]
    shap_df = payload["shap_df"]
    base_value = payload["base_value"]
    xai_agg = payload["xai_agg"]

    st.subheader("Recommended pizza")

    dietary_note = ""
    if payload["inputs"]["dietary_restriction"] == "Other (please specify)" and payload["inputs"]["dietary_restriction_other_text"].strip():
        dietary_note = f"\n- Reported dietary note: {payload['inputs']['dietary_restriction_other_text']}"

    st.success(
        f"**{meta['name']}**\n\n"
        f"- Style: {meta['style']}\n"
        f"- Main ingredient: {meta['ingredient']}\n"
        f"- Dietary tag: {meta['dietary_tag']}\n"
        f"- Customer rating: {meta['customer_rating']}\n"
        f"- Free delivery: {meta['free_delivery']}\n"
        f"- Price: ${meta['price']} CAD"
        f"{dietary_note}"
    )

    if GROUP == "visual":
        st.subheader("Why this pizza was recommended")
        st.caption("This visual explanation shows which parts of your input pushed the model toward this recommendation and which pushed it away.")
        fig = plot_shap_waterfall(shap_df, base_value=base_value, max_display=10)
        st.pyplot(fig)
    else:
        st.subheader("Why this pizza was recommended")
        st.caption("This text explanation summarizes the most influential factors in the recommendation.")
        top_features = xai_agg["study_feature"].tolist()[:3]
        st.write(
            f"The recommendation was influenced mostly by **{top_features[0]}**, "
            f"then **{top_features[1]}**, and then **{top_features[2]}**."
        )

    st.subheader("AI feature importance summary")
    st.dataframe(
        xai_agg.rename(columns={
            "study_feature": "Feature",
            "importance": "XAI importance",
            "xai_rank": "XAI rank",
        }),
        use_container_width=True,
        hide_index=True,
    )

    return_url = build_return_url(payload)
    st.link_button("Continue to Survey", return_url, use_container_width=True)

else:
    st.caption("Complete all sections and click Get recommendation to continue.")