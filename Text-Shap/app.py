import uuid
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

import shap

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(page_title="Pizza (SHAP) Study", layout="centered")

# -----------------------------
# Qualtrics link
# -----------------------------
QUALTRICS_BASE_URL = "https://concordia.yul1.qualtrics.com/jfe/form/SV_6DMC76bV1o5YVPE"

# -----------------------------
# Pizza catalog (small fixed menu)
# -----------------------------
PIZZAS = [
    {"pizza_id": "MARGHERITA", "name": "Margherita",  "style": "Italian",  "veg": "Veg",  "spicy": "No",  "price": 12},
    {"pizza_id": "PEPPERONI",  "name": "Pepperoni",   "style": "American", "veg": "Meat", "spicy": "No",  "price": 14},
    {"pizza_id": "DIAVOLA",    "name": "Diavola",     "style": "Italian",  "veg": "Meat", "spicy": "Yes", "price": 16},
    {"pizza_id": "VEGGIE",     "name": "Veggie",      "style": "American", "veg": "Veg",  "spicy": "No",  "price": 13},
    {"pizza_id": "BBQ_CHICKEN","name": "BBQ Chicken", "style": "American", "veg": "Meat", "spicy": "No",  "price": 17},
    {"pizza_id": "SPICY_VEG",  "name": "Spicy Veg",   "style": "Italian",  "veg": "Veg",  "spicy": "Yes", "price": 15},
]
PIZZA_ID_TO_META = {p["pizza_id"]: p for p in PIZZAS}

CAT_FEATURES = ["diet", "spicy", "style", "budget_sensitivity"]
NUM_FEATURES = ["max_price"]

# -----------------------------
# Synthetic training data generator
# -----------------------------
def generate_synthetic_training_data(n=2500, seed=7):
    """
    Generates synthetic user preference profiles and assigns a pizza label using a scoring rule.
    This is a *simulation* to create a controllable model for the experiment.
    """
    rng = np.random.default_rng(seed)
    max_price = rng.integers(10, 26, size=n)
    diet = rng.choice(["Veg", "Meat"], size=n, p=[0.45, 0.55])
    spicy = rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65])
    style = rng.choice(["Italian", "American"], size=n, p=[0.55, 0.45])
    budget_sensitivity = rng.choice(["High", "Medium", "Low"], size=n, p=[0.35, 0.45, 0.20])

    labels = []
    for i in range(n):
        mp = int(max_price[i])
        d = diet[i]
        sp = spicy[i]
        stl = style[i]
        bs = budget_sensitivity[i]

        affordable = [p for p in PIZZAS if p["price"] <= mp]
        if not affordable:
            affordable = sorted(PIZZAS, key=lambda x: x["price"])[:1]

        def score(p):
            s = 0.0
            s += 2.0 if p["veg"] == d else 0.0
            s += 2.0 if p["spicy"] == sp else 0.0
            s += 2.0 if p["style"] == stl else 0.0
            if bs == "High":
                s -= 0.35 * p["price"]
            elif bs == "Medium":
                s -= 0.20 * p["price"]
            else:
                s -= 0.10 * p["price"]
            s += rng.normal(0, 0.3)  # noise for realism
            return s

        best = max(affordable, key=score)
        labels.append(best["pizza_id"])

    X = pd.DataFrame({
        "max_price": max_price,
        "diet": diet,
        "spicy": spicy,
        "style": style,
        "budget_sensitivity": budget_sensitivity
    })
    y = pd.Series(labels, name="pizza_id")
    return X, y

# -----------------------------
# Train model + SHAP explainer
# -----------------------------
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

    clf = LogisticRegression(max_iter=3000, multi_class="auto")
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y)

    feature_names = pipe.named_steps["pre"].get_feature_names_out().tolist()
    feature_names = [n.replace("cat__", "").replace("num__", "") for n in feature_names]

    # background sample for SHAP baseline
    X_bg = X.sample(n=min(300, len(X)), random_state=42)
    X_bg_trans = pipe.named_steps["pre"].transform(X_bg)

    explainer = shap.LinearExplainer(
        pipe.named_steps["clf"],
        X_bg_trans,
        feature_perturbation="interventional",
    )
    return pipe, explainer, feature_names

model, explainer, FEATURE_NAMES = train_model_and_explainer()

def _to_dense_1d(mat):
    if hasattr(mat, "toarray"):
        return np.asarray(mat.toarray()).ravel()
    return np.asarray(mat).ravel()

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
    df = pd.DataFrame({
        "feature": FEATURE_NAMES[:n],
        "value": x_vec[:n],
        "shap_value": sv[:n],
    })

    # keep active one-hot features + numeric
    is_numeric = df["feature"].isin(NUM_FEATURES)
    active = (df["value"] != 0) | is_numeric
    df = df[active].copy()

    df["abs"] = df["shap_value"].abs()
    df = df.sort_values("abs", ascending=False).drop(columns=["abs"])
    return pred_class, bv, df

# -----------------------------
# NEW: turn SHAP table into a text explanation
# -----------------------------
def decode_onehot_to_human(df: pd.DataFrame):
    """
    Converts one-hot feature names like 'diet_Veg' into:
    - group='diet'
    - level='Veg'
    For numeric feature 'max_price' => group='max_price', level=None
    """
    rows = []
    for _, r in df.iterrows():
        feat = str(r["feature"])
        val = float(r["value"])
        sv = float(r["shap_value"])

        if feat in NUM_FEATURES:
            rows.append({"group": feat, "level": None, "value": val, "shap_value": sv})
            continue

        # one-hot pattern: "<group>_<level>"
        if "_" in feat:
            g, lvl = feat.split("_", 1)
        else:
            g, lvl = feat, None

        rows.append({"group": g, "level": lvl, "value": val, "shap_value": sv})
    return pd.DataFrame(rows)

def format_shap_text_explanation(
    shap_df: pd.DataFrame,
    predicted_id: str,
    pizza_name: str,
    top_k: int = 6,
):
    """
    Creates a readable text explanation:
    - shows strongest positive drivers and strongest negative drivers
    """
    decoded = decode_onehot_to_human(shap_df)

    # Aggregate by group+level (already unique for one-hot, but safe)
    agg = decoded.groupby(["group", "level"], dropna=False, as_index=False)["shap_value"].sum()

    pos = agg.sort_values("shap_value", ascending=False).head(top_k)
    neg = agg.sort_values("shap_value", ascending=True).head(top_k)

    def pretty_row(row):
        g = row["group"]
        lvl = row["level"]
        sv = row["shap_value"]
        if pd.isna(lvl) or lvl is None:
            # numeric
            return f"- {g}: contribution {sv:+.3f}"
        return f"- {g} = {lvl}: contribution {sv:+.3f}"

    lines = []
    lines.append(f"Model recommended **{pizza_name}** (class = `{predicted_id}`).")
    lines.append("")
    lines.append("**Top factors pushing TOWARD this recommendation (positive SHAP):**")
    if len(pos) == 0:
        lines.append("- (none)")
    else:
        for _, r in pos.iterrows():
            lines.append(pretty_row(r))

    lines.append("")
    lines.append("**Top factors pushing AGAINST this recommendation (negative SHAP):**")
    if len(neg) == 0:
        lines.append("- (none)")
    else:
        for _, r in neg.iterrows():
            lines.append(pretty_row(r))

    lines.append("")
    lines.append("Interpretation rule:")
    lines.append("- Positive values support the recommended pizza; negative values oppose it.")
    lines.append("- The magnitude shows how strong the effect was for THIS user input.")
    return "\n".join(lines)

def build_qualtrics_link(payload: dict) -> str:
    params = {
        "pid": payload["participant_id"],
        "cond": payload["condition"],
        "task": "Pizza",
        "rec_id": payload["recommended_pizza_id"],
        "rec_name": payload["recommended_pizza_name"],
        "rec_price": payload["recommended_pizza_price"],
        "max_price": payload["inputs"]["max_price"],
        "diet": payload["inputs"]["diet"],
        "spicy": payload["inputs"]["spicy"],
        "style": payload["inputs"]["style"],
        "budget_sensitivity": payload["inputs"]["budget_sensitivity"],
        "ts": payload["timestamp"],
    }
    return f"{QUALTRICS_BASE_URL}?{urlencode(params)}"

# -----------------------------
# Session state
# -----------------------------
if "participant_id" not in st.session_state:
    st.session_state.participant_id = str(uuid.uuid4())[:8]

# condition label changed to Text (so you pass it to Qualtrics correctly)
st.session_state.condition = "Text"

if "result_ready" not in st.session_state:
    st.session_state.result_ready = False
if "result_payload" not in st.session_state:
    st.session_state.result_payload = None

# -----------------------------
# UI text improvements
# -----------------------------
DIET_UI_TO_MODEL = {"Vegetarian": "Veg", "Non-vegetarian": "Meat"}
DIET_MODEL_TO_UI = {"Veg": "Vegetarian", "Meat": "Non-vegetarian"}

SPICY_UI_TO_MODEL = {"Not spicy": "No", "Spicy": "Yes"}
SPICY_MODEL_TO_UI = {"No": "Not spicy", "Yes": "Spicy"}

BUDGET_UI_TO_MODEL = {
    "High (price matters a lot)": "High",
    "Medium": "Medium",
    "Low (price matters less)": "Low"
}
BUDGET_MODEL_TO_UI = {v: k for k, v in BUDGET_UI_TO_MODEL.items()}

# -----------------------------
# Single-page app
# -----------------------------
st.title("🍕 Pizza Recommendation")
st.caption(
    "Enter your preferences, receive an AI recommendation, view a text explanation (SHAP), "
    "and then complete a short survey in Qualtrics."
)

st.divider()

st.subheader("Your preferences")

max_price = st.slider(
    "What is the maximum price you are willing to pay (USD)?",
    min_value=10, max_value=25, value=15, step=1
)

diet_ui = st.radio(
    "Which option best describes your dietary preference for this order?",
    options=list(DIET_UI_TO_MODEL.keys()),
    horizontal=True
)
diet = DIET_UI_TO_MODEL[diet_ui]

spicy_ui = st.radio(
    "How spicy would you like your pizza to be?",
    options=list(SPICY_UI_TO_MODEL.keys()),
    horizontal=True
)
spicy = SPICY_UI_TO_MODEL[spicy_ui]

style = st.radio(
    "Which pizza style do you prefer?",
    options=["Italian", "American"],
    horizontal=True
)

budget_ui = st.selectbox(
    "How sensitive are you to price when choosing a pizza?",
    options=list(BUDGET_UI_TO_MODEL.keys()),
    index=1
)
budget_sensitivity = BUDGET_UI_TO_MODEL[budget_ui]

x = pd.DataFrame([{
    "max_price": max_price,
    "diet": diet,
    "spicy": spicy,
    "style": style,
    "budget_sensitivity": budget_sensitivity
}])

st.divider()

if st.button("Get recommendation", type="primary"):
    pred_id = model.predict(x)[0]
    meta = PIZZA_ID_TO_META[pred_id]

    # Ensure recommendation respects budget (hard constraint)
    if meta["price"] > max_price:
        affordable = [p for p in PIZZAS if p["price"] <= max_price]
        if affordable:
            def sim(p):
                s = 0
                s += 1 if p["veg"] == diet else 0
                s += 1 if p["spicy"] == spicy else 0
                s += 1 if p["style"] == style else 0
                if budget_sensitivity == "High":
                    s += (25 - p["price"]) / 25
                return s
            meta = max(affordable, key=sim)
            pred_id = meta["pizza_id"]

    _, base_value, shap_df = compute_shap_for_row(model, explainer, x)

    # NEW: build text explanation now (store in payload)
    shap_text = format_shap_text_explanation(
        shap_df=shap_df,
        predicted_id=pred_id,
        pizza_name=meta["name"],
        top_k=6,
    )

    st.session_state.result_ready = True
    st.session_state.result_payload = {
        "timestamp": int(time.time()),
        "participant_id": st.session_state.participant_id,
        "condition": st.session_state.condition,
        "inputs": x.iloc[0].to_dict(),
        "recommended_pizza_id": pred_id,
        "recommended_pizza_name": meta["name"],
        "recommended_pizza_price": meta["price"],
        "meta": meta,
        "shap_df": shap_df,
        "base_value": base_value,
        "shap_text": shap_text,  # NEW
    }
    st.rerun()

# -----------------------------
# Results + explanation + Qualtrics
# -----------------------------
if st.session_state.result_ready and st.session_state.result_payload is not None:
    payload = st.session_state.result_payload
    meta = payload["meta"]

    st.divider()
    st.subheader("Result")

    st.success(
        f"**Recommended pizza:** {meta['name']}  \n"
        f"**Style:** {meta['style']}  \n"
        f"**Diet:** {DIET_MODEL_TO_UI.get(meta['veg'], meta['veg'])}  \n"
        f"**Spice level:** {SPICY_MODEL_TO_UI.get(meta['spicy'], meta['spicy'])}  \n"
        f"**Price:** ${meta['price']}"
    )

    st.subheader("Text explanation")
    st.caption(
        "This explanation lists the main inputs that increased or decreased the model’s score for the recommended pizza."
    )
    st.markdown(payload["shap_text"])

    # Optional: show the raw SHAP table for debugging / transparency
    with st.expander("Show raw SHAP table (optional)"):
        st.dataframe(payload["shap_df"], use_container_width=True)

    st.subheader("Survey (Qualtrics)")
    st.caption("Click below to open the survey. Your responses will be stored in Qualtrics.")

    qualtrics_link = build_qualtrics_link(payload)
    st.link_button("Open Qualtrics Survey", qualtrics_link)

    st.caption(f"Participant ID: **{payload['participant_id']}** (passed to Qualtrics as pid)")

    if st.button("Start over (new participant)"):
        st.session_state.participant_id = str(uuid.uuid4())[:8]
        st.session_state.result_ready = False
        st.session_state.result_payload = None
        st.rerun()
else:
    st.caption("Complete and click **Get recommendation** to continue.")