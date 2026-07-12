import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import os

# Configuración de página
st.set_page_config(layout="wide", page_title="Q-SOC Parking Demo")
st.title("Sistema Inteligente de Estacionamientos (Q-SOC)")

# Cargar el modelo (asegúrate de que 'best.pt' esté en la raíz)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

modelo = load_model()




# --- INTERFAZ PERSONALIZADA ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualización del Mapa")
    # Mostrar la imagen con las cajas de Ultralytics (o puedes dibujar las tuyas encima)
    st.image(resultado.plot(), channels="BGR", use_column_width=True)
with col2:
    # Selección de imagen
    img_files = ['ref_images/im01.jpg', 'ref_images/im02.jpg', 'ref_images/im03.jpg','ref_images/im04.jpg','ref_images/im05.jpg','ref_images/im06.jpg']
    selected_img = st.selectbox("Selecciona una imagen de prueba:", img_files)
    st.subheader("Estado del Recinto")
    
    
    # Ejecutar tu lógica de predicción
    resultados = modelo.predict(source=selected_img, imgsz=960, conf=0.25, verbose=False)
    resultado = resultados[0]
    # Tu lógica de conteo
    detecciones = []
    for box in resultado.boxes:
        detecciones.append(modelo.names[int(box.cls[0])])
    
    df = pd.Series(detecciones).value_counts()
    free_count = int(df.get("free_space", 0))
    occupied_count = int(df.get("occupied_space", 0))
    
    # Tarjetas visuales
    st.metric("Espacios Libres", free_count)
    st.metric("Espacios Ocupados", occupied_count)
    
# ... (código anterior igual)

st.divider()
st.subheader("Dataset de Referencia (Imágenes Originales)")

# Buscamos todas las imágenes en la carpeta 'ref_images'
ruta_carpeta = 'ref_images'
imagenes_en_carpeta = [f for f in os.listdir(ruta_carpeta) if f.endswith('.jpg')]

# Creamos las columnas dinámicamente según la cantidad de imágenes
cols_ref = st.columns(len(imagenes_en_carpeta))

for i, img_name in enumerate(imagenes_en_carpeta):
    # Esto agrega cada imagen encontrada automáticamente
    ruta_completa = os.path.join(ruta_carpeta, img_name)
    cols_ref[i].image(ruta_completa, use_column_width=True, caption=img_name)