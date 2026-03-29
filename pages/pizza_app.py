from urllib.parse import urlencode
import streamlit as st

st.set_page_config(page_title="Pizza App", layout="centered")

qp = st.query_params

def q(name: str) -> str:
    value = qp.get(name, "")
    if isinstance(value, list):
        return value[0] if value else ""
    return str(value).strip()

PID = q("pid")
GROUP = q("group")
APP1 = q("app1")
APP2 = q("app2")
APP3 = q("app3")
STEP = q("step")
APP_NAME = q("app")

POST_SURVEY_MAP = {
    "1": "https://concordia.yul1.qualtrics.com/jfe/form/SV_2cps8pBYmqBoJxk",
    "2": "https://concordia.yul1.qualtrics.com/jfe/form/SV_SURVEY2_ID",
    "3": "https://concordia.yul1.qualtrics.com/jfe/form/SV_SURVEY3_ID",
}

st.title("🍕 Pizza App Test")
st.write("This is the first app test page.")
st.write(f"Participant ID: {PID}")
st.write(f"Group: {GROUP}")
st.write(f"Step: {STEP}")
st.write(f"App: {APP_NAME}")

max_price = st.slider("Maximum price", 15, 50, 25)
pizza_style = st.radio("Pizza style", ["Italian", "American"], horizontal=True)
ingredient = st.selectbox("Preferred ingredient", ["Cheese", "Pepperoni", "Chicken", "Vegetables", "Mushrooms"])

st.subheader("Recommendation")
st.success("Recommended pizza: Margherita")

def build_return_url():
    base_url = POST_SURVEY_MAP[STEP]
    params = {
        "pid": PID,
        "group": GROUP,
        "app1": APP1,
        "app2": APP2,
        "app3": APP3,
        "step": STEP,
        "current_app": APP_NAME,
        "task": "pizza",
        "rec_id": "MARGHERITA",
        "rec_name": "Margherita",
        "rec_price": 18,
        "max_price": max_price,
        "pizza_style": pizza_style,
        "ingredient_preference": ingredient,
    }
    return f"{base_url}?{urlencode(params)}"

return_url = build_return_url()
st.link_button("Continue to Survey", return_url, use_container_width=True)