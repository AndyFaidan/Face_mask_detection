from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_webcam

# mengatur tata letak halaman
st.set_page_config(
    page_title="Deteksi Masker Wajah untuk YOstreLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# judul utama halaman
st.title("Deteksi Masker Wajah")

# sidebar
st.sidebar.header("Konfigurasi Model DL")

# opsi model
task_type = st.sidebar.selectbox(
    "Pilih Tugas",
    ["Deteksi"]
)

model_type = None
if task_type == "Deteksi":
    model_type = st.sidebar.selectbox(
        "Pilih Model",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Saat ini hanya fungsi 'Deteksi' yang diimplementasikan")

confidence = float(st.sidebar.slider(
    "Pilih Tingkat Keyakinan Model", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Silakan Pilih Model di Sidebar")

# memuat model DL yang telah dilatih
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Tidak dapat memuat model. Harap periksa path yang ditentukan: {model_path}")

# opsi gambar/video
st.sidebar.header("Konfigurasi Gambar/Video")
source_selectbox = st.sidebar.selectbox(
    "Pilih Sumber",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Gambar
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Saat ini hanya sumber 'Gambar' dan 'Video' yang diimplementasikan")
