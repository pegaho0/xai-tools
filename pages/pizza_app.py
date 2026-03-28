import streamlit as st

st.set_page_config(page_title="Pizza Test", layout="centered")

st.title("Pizza App Test")
st.success("Pizza page loaded successfully.")
st.write("This is a minimal deploy-safe test page.")
st.write("Query params:", dict(st.query_params))