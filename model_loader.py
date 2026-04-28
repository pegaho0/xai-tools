from pathlib import Path
import joblib
import gdown
import streamlit as st
from gdown.exceptions import FileURLRetrievalError

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_FILE_IDS = {
    "pizza": "1f-3vsRGPHI869LPjpj6uYDVJKI-xCXQo",
    "tour": "1rMZGC8tDHz4OTwUorqiT9SsRN05QszYm",
    "house": "1asO_dOsfbMMDUMcuOcFOVXyeKTBhDj7G",
}

MODEL_PATHS = {
    "pizza": MODELS_DIR / "pizza_bundle.joblib",
    "tour": MODELS_DIR / "tour_bundle.joblib",
    "house": MODELS_DIR / "house_bundle.joblib",
}


def download_model(task_name):
    file_id = MODEL_FILE_IDS[task_name]
    output_path = MODEL_PATHS[task_name]
    urls = [
        f"https://drive.google.com/uc?id={file_id}",
        f"https://drive.google.com/uc?export=download&id={file_id}",
        f"https://drive.google.com/file/d/{file_id}/view?usp=sharing",
    ]

    last_error = None
    for url in urls:
        try:
            downloaded = gdown.download(
                url=url,
                output=str(output_path),
                quiet=False,
                fuzzy=True,
                use_cookies=False,
            )
            if downloaded and output_path.exists() and output_path.stat().st_size > 0:
                return
        except Exception as exc:
            last_error = exc

    # If all attempts failed, raise a clear error for caller.
    if isinstance(last_error, FileURLRetrievalError):
        raise RuntimeError(
            f"Could not download '{task_name}' model from Google Drive. "
            "Please verify the file is publicly accessible (Anyone with the link) "
            "and has not exceeded download quota."
        ) from last_error
    raise RuntimeError(f"Could not download '{task_name}' model.") from last_error


@st.cache_resource
def load_model_bundle(task_name):
    model_path = MODEL_PATHS[task_name]

    if not model_path.exists():
        with st.spinner(f"Downloading {task_name} model..."):
            try:
                download_model(task_name)
            except Exception as exc:
                st.error(
                    "Model download failed. Please ask the study admin to verify Google Drive sharing "
                    "and file download quota for this model."
                )
                st.caption(f"Technical detail: {exc}")
                st.stop()

    return joblib.load(model_path)