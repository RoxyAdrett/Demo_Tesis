import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np

# Configuración de página
st.set_page_config(layout="wide", page_title="Q-SOC Parking Demo")
st.title("Sistema Inteligente de Estacionamientos (Q-SOC)")

@st.cache_resource
def load_model():
    return YOLO('best.pt')

modelo = load_model()

# --- SELECTOR GLOBAL (Arriba de todo) ---
ruta_carpeta = 'ref_images'
imagenes = sorted([f for f in os.listdir(ruta_carpeta) if f.endswith('.jpg')])
selected_img_name = st.selectbox("Entrada de Datos: Selecciona imagen a analizar:", imagenes)
selected_path = os.path.join(ruta_carpeta, selected_img_name)

# --- PREDICCIÓN ---
img = cv2.imread(selected_path)
resultados = modelo.predict(source=img, imgsz=960, conf=0.25, verbose=False)
resultado = resultados[0]

# --- DIBUJO MANUAL ---
img_bgr = img.copy()
for box in resultado.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    color = (0, 255, 0) if cls == 0 else (0, 0, 255) # Verde libre, Rojo ocupado
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)

# --- INTERFAZ PRINCIPAL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualización del Mapa")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

with col2:
    st.subheader("Estado del Recinto")
    with st.container():
        detecciones = [modelo.names[int(box.cls[0])] for box in resultado.boxes]
        df = pd.Series(detecciones).value_counts()
        
        libres = int(df.get("free_space", 0))
        ocupados = int(df.get("occupied_space", 0))
        total = libres + ocupados

        st.metric("✅ Espacios Libres", libres)
        st.metric("❌ Espacios Ocupados", ocupados)
        st.metric("🅿️ Total de Espacios", total)

# --- GALERÍA INFERIOR ---
st.divider()
st.subheader("Dataset de Referencia")

n_cols = 4 # Número de imágenes por fila en la galería
cols_galeria = st.columns(n_cols)
for i, img_name in enumerate(imagenes):
    with cols_galeria[i % n_cols]:
        st.image(os.path.join(ruta_carpeta, img_name), caption=img_name, use_column_width=True)

# --- VIDEO ---
st.divider()
st.subheader("Demostración de Procesamiento en Video")
if os.path.exists('demo_video.mp4'):
    st.video('demo_video.mp4')
else:
    st.warning("El video 'demo_video.mp4' no se encuentra en la raíz del repositorio.")