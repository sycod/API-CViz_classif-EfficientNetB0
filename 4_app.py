"""Streamlit simple app to predict dog breed from an image"""

import os
import logging
import streamlit as st
import dill as pickle

# home made
# from src.app_utils import xxxxxxxxx


# CONFIG
# logging configuration (see all outputs, even DEBUG or INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# update session state on inputs
# def update_title():
#     st.session_state.title_input = st.session_state.title


# def update_body():
#     st.session_state.body_input = st.session_state.body


# main function, triggered with new user upload
def on_upload():
    """Actions to perform when user uploads images"""
    prediction = "Upload ok!"
    for i, img in enumerate(user_input):
        st.write(f"image {i+1}: {prediction}")

#     user_input = st.session_state.title_input + "\n" + st.session_state.body_input
#     logging.info(f"\nUser input: {user_input}")


#     # check user input length
#     if not check_length(user_input):
#         logging.warning(f"⚠️  Input length is too short")
#         st.session_state.predicted_tags = None
#         st.session_state.message = "⚠️  Input length is too short"
#     else:
#         # preprocess input
#         input_clean = preprocess_doc(
#             user_input, st.session_state.keep_set, st.session_state.exclude_set
#         )
#         logging.info(f"\nClean input: {input_clean}")

#         # check preprocessed input length before predict
#         if not check_length(input_clean):
#             logging.warning(f"⚠️  Length is too short after preprocessing: check input")
#             st.session_state.predicted_tags = None
#             st.session_state.message = (
#                 "⚠️  Length is too short after preprocessing: check input"
#             )
#         else:
#             # predict tags
#             predicted_tags = predict_tags(
#                 input_clean, st.session_state.vectorizer, st.session_state.classifier
#             )
#             st.session_state.predicted_tags = predicted_tags

#         # log infos
#         logging.info(f"\nPredicted tags: {st.session_state.predicted_tags}")

#     return st.session_state.predicted_tags


def launch_api():
    """Launch API server"""
    # # CHECK ML TOOLS & SETUP SESSION STATE
    # # load keep set (for preprocessing)
    # if "keep_set" not in st.session_state:
    #     logging.info(f"⚙️  Loading keep set...")
    #     if os.path.exists(KEEP_SET_URI):
    #         with open(KEEP_SET_URI, "rb") as f:
    #             keep_set = pickle.load(f)
    #         st.session_state.keep_set = keep_set
    #         logging.info(f"✅ Keep set loaded")
    #     else:
    #         logging.warning(f"⚠️ No keep set found ⚠️")
    # # load exclude set (for preprocessing)
    # if "exclude_set" not in st.session_state:
    #     logging.info(f"⚙️  Loading exclude set...")
    #     if os.path.exists(EXCLUDE_SET_URI):
    #         with open(EXCLUDE_SET_URI, "rb") as f:
    #             exclude_set = pickle.load(f)
    #         st.session_state.exclude_set = exclude_set
    #         logging.info(f"✅ Exclude set loaded")
    #     else:
    #         logging.warning(f"⚠️ No exclude set found ⚠️")
    # # load vectorizer
    # if "vectorizer" not in st.session_state:
    #     logging.info(f"⚙️  Loading vectorizer...")
    #     if os.path.exists(VECTORIZER_URI):
    #         vectorizer = Word2Vec.load(VECTORIZER_URI)
    #         st.session_state.vectorizer = vectorizer
    #         logging.info(f"✅ Vectorizer loaded")
    #     else:
    #         logging.warning(f"⚠️ No vectorizer found ⚠️")
    # # load classifier
    # if "classifier" not in st.session_state:
    #     logging.info(f"⚙️  Loading classifier...")
    #     if os.path.exists(CLASSIFIER_URI):
    #         with open(CLASSIFIER_URI, "rb") as f:
    #             classifier = pickle.load(f)
    #         st.session_state.classifier = classifier
    #         logging.info(f"✅ Classifier loaded")
    #     else:
    #         logging.warning(f"⚠️ No classifier found ⚠️")
    # # placeholders (if not in session state)
    # if "title_input" not in st.session_state:
    #     st.session_state.title_input = ""
    # if "body_input" not in st.session_state:
    #     st.session_state.body_input = ""
    # if "predicted_tags" not in st.session_state:
    #     st.session_state.predicted_tags = TAGS_PLACEHOLDER
    # if "message" not in st.session_state:
    #     st.session_state.message = None

    # GUI
    st.set_page_config(
        page_title="Which breed is that dog?",
        page_icon="api_favicon.ico",
        layout="centered",
    )
    st.write("# Send dog images to search among 120 dog breeds")
    st.write("💡 Model trained upon the Stanford Dogs Dataset")
    st.write("1️⃣ For better results, use 1 dog per image")
    st.write("*️⃣ Multiple files uploads are accepted")
    st.write("⚠️ Only JPG and PNG files allowed, max upload: 200MB")

    # user input
    user_input = st.file_uploader(
        "Send your dog(s)",
        accept_multiple_files=True,
        type=['png', 'jpg', 'jpeg']
        # key="user_input",
        on_change=on_upload,
    )

    # # predictions
    # st.button(
    #     "⬇️  Predict tags  ⬇️",
    #     type="primary",
    #     use_container_width=True,
    #     on_click=click_button,
    # )
    # # display message if no prediction (e.g. input is too short)
    # if st.session_state.predicted_tags is not None:
    #     st.write("#### :blue[{}]".format(st.session_state.predicted_tags))
    # else:
    #     st.write("#### :red[{}]".format(st.session_state.message))


if __name__ == "__main__":
    launch_api()
