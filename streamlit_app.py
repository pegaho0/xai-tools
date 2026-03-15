import streamlit as st

st.set_page_config(page_title="Pizza Study Controller", layout="centered")

# -----------------------------
# Config
# -----------------------------
PAGE_MAP = {
    ("visual", "app1"): "pages/10_visual_app1.py",
    ("visual", "app2"): "pages/20_visual_app2.py",
    ("visual", "app3"): "pages/30_visual_app3.py",
    ("textual", "app1"): "pages/40_textual_app1.py",
    ("textual", "app2"): "pages/50_textual_app2.py",
    ("textual", "app3"): "pages/60_textual_app3.py",
}

# -----------------------------
# Read query params
# Example:
# ?pid=R_ABC123&group=visual&step=app1&rid=RSP_xxx
# -----------------------------
qp = st.query_params

pid = qp.get("pid", "")
group = qp.get("group", "")
step = qp.get("step", "")
rid = qp.get("rid", "")

st.title("Experiment Controller")

# -----------------------------
# Validation
# -----------------------------
if not pid:
    st.error("Missing participant id (pid). Come here only from Qualtrics.")
    st.stop()

if group not in ["visual", "textual"]:
    st.error("Missing or invalid group. Expected 'visual' or 'textual'.")
    st.stop()

if step not in ["app1", "app2", "app3"]:
    st.error("Missing or invalid step. Expected app1, app2, or app3.")
    st.stop()

target_page = PAGE_MAP.get((group, step))
if not target_page:
    st.error("No page found for this condition.")
    st.stop()

st.info(f"Participant: {pid} | Group: {group} | Step: {step}")

# -----------------------------
# Auto route
# -----------------------------
st.success("Routing you to the correct app...")
st.switch_page(
    target_page,
    query_params={
        "pid": pid,
        "group": group,
        "step": step,
        "rid": rid,
    },
)
