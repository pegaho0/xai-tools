import streamlit as st

st.set_page_config(page_title="Study Controller", layout="centered")

qp = st.query_params

def q(name: str) -> str:
    value = qp.get(name, "")
    if isinstance(value, list):
        return value[0] if value else ""
    return str(value).strip()

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

st.title("Experiment Controller")

errors = []

if not pid:
    errors.append("Missing pid")
if group not in VALID_GROUPS:
    errors.append("Invalid group. Expected visual or text")
if app1 not in VALID_APPS:
    errors.append("Invalid app1")
if app2 not in VALID_APPS:
    errors.append("Invalid app2")
if app3 not in VALID_APPS:
    errors.append("Invalid app3")
if step not in VALID_STEPS:
    errors.append("Invalid step. Expected 1, 2, or 3")
if app not in VALID_APPS:
    errors.append("Invalid app")

expected_app = {"1": app1, "2": app2, "3": app3}.get(step)
if expected_app and app != expected_app:
    errors.append(f"Step/app mismatch. step={step} should use {expected_app}, but got {app}")

if errors:
    st.error("Routing error. Please start from Qualtrics.")
    for e in errors:
        st.write(f"- {e}")
    st.stop()

st.success("Routing to the correct app...")

# فعلاً فقط app_a = pizza app
if app == "app_a":
    st.switch_page("pages/10_pizza_app.py")
else:
    st.error(
        f"{app} is not implemented yet. "
        f"For now, only app_a is connected."
    )
    st.stop()