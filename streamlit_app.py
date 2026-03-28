import streamlit as st
import streamlit.components.v1 as components
from urllib.parse import urlencode

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

# فعلاً فقط app_a را وصل کرده‌ایم
if app == "app_a":
    params = urlencode({
        "pid": pid,
        "group": group,
        "app1": app1,
        "app2": app2,
        "app3": app3,
        "step": step,
        "app": app,
    })
    target_url = f"/pizza_app?{params}"

    st.success("Routing to pizza app...")

    components.html(
        f"""
        <script>
            window.top.location.href = "{target_url}";
        </script>
        """,
        height=0,
    )
    st.stop()
else:
    st.error(f"{app} is not implemented yet. For now, only app_a is connected.")
    st.stop()