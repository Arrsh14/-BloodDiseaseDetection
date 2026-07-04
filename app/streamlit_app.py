"""
Streamlit demo app for the Blood Disease Detection system.
Upload a blood smear image + enter CBC lab values, get a fused diagnosis
combining tabular (XGBoost) and CNN (leukemia/malaria) model outputs.
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference.predict import predict, _leukemia_model, _malaria_model, _device
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from src.explainability.gradcam_leukemia import denormalize
from src.data.preprocess_images import get_eval_transforms
import torch


st.set_page_config(page_title="Blood Disease Detection", layout="centered", page_icon="🩸")

# --- Light theme + card styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #f7f8fa;
        color: #1a1a1a;
    }
    .block-container {
        padding-top: 2.5rem;
        max-width: 850px;
    }
    h1, h2, h3 {
        color: #1a1a1a !important;
        font-weight: 700 !important;
    }
    p, span, label, .stMarkdown {
        color: #333333;
    }
    .card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 1.75rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        border: 1px solid #ececec;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stFileUploader"] {
        background-color: #fafbfc;
        border-radius: 12px;
        padding: 0.5rem;
    }
    .stButton > button {
        background-color: #e63946;
        color: white;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        border: none;
        transition: 0.2s;
    }
    .stButton > button:hover {
        background-color: #c1121f;
        color: white;
    }
    div[data-testid="stMetric"] {
        background-color: #fafbfc;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #ececec;
    }
    .stAlert {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🩸 Blood Disease Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; color:#666; font-size:0.95rem;'>"
    "Upload a blood smear image and enter CBC lab values to get a fused diagnosis "
    "combining a tabular model and two CNNs.<br>"
    "<b>Note:</b> this uses synthetic tabular training data and is a portfolio/research "
    "project — not a diagnostic tool.</p>",
    unsafe_allow_html=True,
)

st.write("")

# --- Input section ---
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("1. Blood Smear Image")
    uploaded_image = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("2. CBC Lab Values")
    wbc_count = st.number_input("WBC count (×10³/µL)", min_value=0.0, value=7.5, step=0.5)
    hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, value=14.0, step=0.1)
    platelet_count = st.number_input("Platelet count (×10³/µL)", min_value=0.0, value=300.0, step=10.0)
    rbc_count = st.number_input("RBC count (million/µL)", min_value=0.0, value=5.0, step=0.1)
    parasitemia_pct = st.number_input("Parasitemia (%)", min_value=0.0, value=0.0, step=0.1)
    st.markdown('</div>', unsafe_allow_html=True)

predict_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

if predict_clicked:
    if not uploaded_image:
        st.error("Please upload a blood smear image before predicting.")
    else:
        with st.spinner("Running models..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                image = Image.open(uploaded_image).convert("RGB")
                image.save(tmp.name)
                tmp_path = tmp.name

            lab_values = {
                "wbc_count": wbc_count,
                "hemoglobin": hemoglobin,
                "platelet_count": platelet_count,
                "rbc_count": rbc_count,
                "parasitemia_pct": parasitemia_pct,
            }

            try:
                result = predict(image_path=tmp_path, lab_values=lab_values)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        # --- Results section ---
        st.write("")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Result")

        diagnosis = result["diagnosis"]
        confidence = result["tabular_confidence"]

        diagnosis_colors = {
            "normal": "green",
            "leukemia": "red",
            "malaria": "orange",
            "both": "violet",
        }
        color = diagnosis_colors.get(diagnosis, "blue")

        st.markdown(f"### Diagnosis: :{color}[{diagnosis.upper()}]")
        st.progress(confidence, text=f"Tabular model confidence: {confidence:.1%}")
        st.info(result["explanation"])
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Probability breakdown ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Tabular Model — Full Probability Breakdown")
        st.bar_chart(result["tabular_probs"])
        st.markdown('</div>', unsafe_allow_html=True)

        # --- CNN confidences ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Leukemia CNN — P(cancer)", f"{result['leukemia_cnn_prob']:.1%}")
        with c2:
            st.metric("Malaria CNN — P(parasitized)", f"{result['malaria_cnn_prob']:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

        # --- Grad-CAM visualization ---
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Grad-CAM: Where the relevant CNN focused")

        relevant_model = _leukemia_model if diagnosis in ("leukemia", "both") else _malaria_model
        model_label = "Leukemia CNN" if diagnosis in ("leukemia", "both") else "Malaria CNN"

        if diagnosis == "normal":
            st.write("No abnormality detected — Grad-CAM shown using leukemia CNN as reference.")
            relevant_model = _leukemia_model
            model_label = "Leukemia CNN"

        eval_transform = get_eval_transforms()
        image_tensor = eval_transform(image)
        input_tensor = image_tensor.unsqueeze(0).to(_device)

        target_layer = relevant_model.layer4[-1]
        cam = GradCAM(model=relevant_model, target_layers=[target_layer])

        with torch.no_grad():
            output = relevant_model(input_tensor)
            pred_class = output.argmax(dim=1).item()

        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred_class)])
        grayscale_cam = grayscale_cam[0, :]

        rgb_img = denormalize(image_tensor.cpu())
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.image(rgb_img, caption="Original", use_container_width=True)
        with col_b:
            st.image(visualization, caption=f"{model_label} attention", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        Path(tmp_path).unlink(missing_ok=True)

st.write("")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.8rem;'>"
    "Built as a portfolio project. Tabular model trained on synthetic CBC data grounded in "
    "published clinical reference ranges (see docs/Dataset_Sources.md). Not intended for real medical use."
    "</p>",
    unsafe_allow_html=True,
)