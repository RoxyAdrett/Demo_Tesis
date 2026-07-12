import streamlit as st
from PIL import Image

st.set_page_config(layout="wide", page_title="Demo Parking AI")
st.title("Sistema Inteligente de Estacionamientos (Q-SOC)")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualización del Modelo")
    # Cambia 'im01.jpg' por la ruta correcta en tu carpeta ref_images
    image = Image.open('ref_images/im01.jpg')
    st.image(image, use_column_width=True)

with col2:
    st.subheader("Estado del Recinto")
    st.metric("Espacios Libres", 12)
    st.metric("Espacios Ocupados", 8)
    st.info("Modelo YOLOv11 en fase de inferencia estática.")

st.divider()
st.subheader("Dataset de Referencia")
cols_ref = st.columns(3)
cols_ref[0].image('ref_images/im01.jpg')
cols_ref[1].image('ref_images/im02.jpg')
cols_ref[2].image('ref_images/im03.jpg')