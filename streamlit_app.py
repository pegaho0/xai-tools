import streamlit as st

from app_core import q

st.set_page_config(page_title="Study Controller", layout="centered")

pid = q("pid")
group = q("group")
app1 = q("app1")
app2 = q("app2")
app3 = q("app3")
step = q("step")
app = q("app")

VALID_GROUPS = {"visual", "text"}
VALID_APPS = {"app_a", "app_b", "app_c"}
VALID_STEPS = {"1", "2", "3"}

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
    errors.append(f"Step/app mismatch. step={step} should use {expected_app}, but got {app}")

if errors:
    st.error("Routing error. Please start from the main study link.")
    st.stop()

st.session_state["pid"] = pid
st.session_state["group"] = group
st.session_state["app1"] = app1
st.session_state["app2"] = app2
st.session_state["app3"] = app3
st.session_state["step"] = step
st.session_state["app"] = app

if app == "app_a":
    st.switch_page("pages/pizza_app.py")
elif app == "app_b":
    st.switch_page("pages/tour_app.py")
elif app == "app_c":
    st.switch_page("pages/house_app.py")
else:
    st.error("Unknown app route.")
    st.stop()