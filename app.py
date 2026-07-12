import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import os
from PIL import Image

# Configuración de página
st.set_page_config(layout="wide", page_title="Q-SOC Parking Demo")
st.title("Sistema Inteligente de Estacionamientos (Q-SOC)")

# Cargar modelo
@st.cache_resource
def load_model():
    return YOLO('best.pt')

modelo = load_model()

# --- SIDEBAR (Costado lateral) ---
with st.sidebar:
    st.header("Configuración")
    ruta_carpeta = 'ref_images'
    # Detectar imágenes automáticamente en la carpeta
    imagenes_en_carpeta = [f for f in os.listdir(ruta_carpeta) if f.endswith('.jpg')]
    selected_img_name = st.selectbox("Selecciona imagen a analizar:", imagenes_en_carpeta)
    selected_path = os.path.join(ruta_carpeta, selected_img_name)

# --- LÓGICA DE PREDICCIÓN ---
resultados = modelo.predict(source=selected_path, imgsz=960, conf=0.25, verbose=False)
resultado = resultados[0]

# --- INTERFAZ PRINCIPAL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualización del Mapa")
    st.image(resultado.plot(), channels="BGR", use_column_width=True)

with col2:
    st.subheader("Estado del Recinto")
    detecciones = [modelo.names[int(box.cls[0])] for box in resultado.boxes]
    df = pd.Series(detecciones).value_counts()
    
    st.metric("Espacios Libres", int(df.get("free_space", 0)))
    st.metric("Espacios Ocupados", int(df.get("occupied_space", 0)))
    
    st.write("---")
    st.markdown("### Detalles Rápidos")
    st.info(f"Archivo analizado: {selected_img_name}")

# --- GALERÍA INFERIOR UNIFORME ---
st.divider()
st.subheader("Dataset de Referencia")

# Definir un tamaño fijo para que todas se vean iguales
TAMANO_FIJO = (300, 200)

cols_ref = st.columns(len(imagenes_en_carpeta))
for i, img_name in enumerate(imagenes_en_carpeta):
    ruta_completa = os.path.join(ruta_carpeta, img_name)
    # Abrir y redimensionar para que todas tengan el mismo aspecto
    img_pil = Image.open(ruta_completa).resize(TAMANO_FIJO)
    cols_ref[i].image(img_pil, use_column_width=True, caption=img_name)