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

# --- SIDEBAR (Selector movido al lateral) ---
with st.sidebar:
    st.header("Entrada de Datos")
    ruta_carpeta = 'ref_images'
    imagenes = [f for f in os.listdir(ruta_carpeta) if f.endswith('.jpg')]
    selected_img_name = st.selectbox("Selecciona imagen a analizar:", imagenes)
    selected_path = os.path.join(ruta_carpeta, selected_img_name)

# --- PREDICCIÓN ---
img = cv2.imread(selected_path)
resultados = modelo.predict(source=img, imgsz=960, conf=0.25, verbose=False)
resultado = resultados[0]

# --- DIBUJO MANUAL (Sin etiquetas de texto) ---
img_bgr = img.copy()
for box in resultado.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    # Verde para libres (0), Rojo para ocupados (1)
    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)

# --- INTERFAZ PRINCIPAL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualización del Mapa")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

with col2:
    st.subheader("Estado del Recinto") # Selector movido arriba de esto
    detecciones = [modelo.names[int(box.cls[0])] for box in resultado.boxes]
    df = pd.Series(detecciones).value_counts()
    
    st.metric("Espacios Libres", int(df.get("free_space", 0)))
    st.metric("Espacios Ocupados", int(df.get("occupied_space", 0)))

# --- GALERÍA INFERIOR ---
st.divider()
st.subheader("Dataset de Referencia")
cols_ref = st.columns(len(imagenes))
for i, img_name in enumerate(imagenes):
    ruta = os.path.join(ruta_carpeta, img_name)
    cols_ref[i].image(Image.open(ruta).resize((300, 200)), use_column_width=True, caption=img_name)