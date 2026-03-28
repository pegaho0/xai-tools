import streamlit as st

st.set_page_config(page_title="Study Controller", layout="centered")

st.title("Deploy Test")
st.success("If you can see this page, Streamlit deployment is working.")
st.write("Next step: reconnect routing and app logic.")
st.write("Query params:", dict(st.query_params))