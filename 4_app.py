"""Streamlit simple app to predict dog breed from an image"""

import os
import yaml
import logging
import streamlit as st


# CONFIG
# local config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
APP_PATH = cfg["app_data"]["local_path"]
WEIGHTS = os.path.join(APP_PATH, cfg["app_data"]["weights"])
BREEDS = cfg["models"]["classes_10"]
# logging configuration (see all outputs, even DEBUG or INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def on_upload():
    """Actions to perform when user uploads images"""
    st.session_state.breed = "wwwwww"


def launch_api():
    """Launch API server"""
    # create session states
    if "image" not in st.session_state:
        st.session_state.image = None
    if "breed" not in st.session_state:
        st.session_state.breed = None

    # load model

    # GUI
    st.set_page_config(
        page_title="Which breed is that dog?",
        page_icon="api_favicon.ico",
        layout="centered",
    )
    st.write("# Send dog image to detect breed among 10")
    st.write(f"Model is trained upon the Stanford Dogs Dataset and able to detect 10 dogs breeds: {(', ').join(BREEDS)}")
    st.markdown("Just **upload your dog image** to predict its breed.")
    st.write("💡 For better results, use 1 dog per image")
    st.write("⚠️ Only JPG and PNG files allowed, max upload: 200MB")
    st.markdown("""---""")

    # user input
    st.session_state.image = st.file_uploader(
        "👇 Upload your dog image 👇",
        accept_multiple_files=False,
        type=['png', 'jpg', 'jpeg'],
        on_change=on_upload,
    )

    st.markdown("""---""")

    if (st.session_state.breed is not None) and (st.session_state.image is not None):
        st.write("#### What a beautiful :blue[{}]:".format(st.session_state.breed))

    if st.session_state.image is not None:
        st.image(st.session_state.image)


if __name__ == "__main__":
    launch_api()
