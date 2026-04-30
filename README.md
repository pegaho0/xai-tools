Fix applied:
- Streamlit was displaying the hybrid explanation HTML as text because the triple-quoted HTML strings were indented.
- Indented HTML in st.markdown can be interpreted as a Markdown code block.
- Added _render_html() helper in app_core.py using textwrap.dedent(...).strip() and unsafe_allow_html=True.
- Updated the hybrid explanation path cards and top SHAP bars to render through _render_html().
