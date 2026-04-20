from pathlib import Path
import joblib
import gdown
import streamlit as st

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_FILE_IDS = {
    "pizza": "1VWSUmz3Ni1X_fncFCGevbewiYOqGFITO",
    "tour": "16XMiBFu461IZWrUu7zga2kY7oglBuK9Y",
    "house": "1yC9L-oQzismXgkqUoWbrRAODSWwvcpZ_",
}

MODEL_PATHS = {
    "pizza": MODELS_DIR / "pizza_bundle.joblib",
    "tour": MODELS_DIR / "tour_bundle.joblib",
    "house": MODELS_DIR / "house_bundle.joblib",
}


def download_model(task_name):
    file_id = MODEL_FILE_IDS[task_name]
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = MODEL_PATHS[task_name]

    gdown.download(url, str(output_path), quiet=False)


@st.cache_resource
def load_model_bundle(task_name):
    model_path = MODEL_PATHS[task_name]

    if not model_path.exists():
        with st.spinner(f"Downloading {task_name} model..."):
            download_model(task_name)

    return joblib.load(model_path)