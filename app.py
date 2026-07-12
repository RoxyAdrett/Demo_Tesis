import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from pathlib import Path
from PIL import Image

# Configuración de página
st.set_page_config(layout="wide", page_title="Q-SOC Parking Demo")
st.title("Sistema Inteligente de Estacionamientos (Q-SOC)")

# Cargar el modelo (asegúrate de que 'best.pt' esté en la raíz)
@st.cache_resource
def load_model():
    return YOLO('best.pt')

modelo = load_model()

# Selección de imagen
img_files = ['im01.jpg', 'im02.jpg', 'im03.jpg']
selected_img = st.selectbox("Selecciona una imagen de prueba:", img_files)

# Ejecutar tu lógica de predicción
resultados = modelo.predict(source=selected_img, imgsz=960, conf=0.25, verbose=False)
resultado = resultados[0]

# --- INTERFAZ PERSONALIZADA ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualización del Mapa")
    # Mostrar la imagen con las cajas de Ultralytics (o puedes dibujar las tuyas encima)
    st.image(resultado.plot(), channels="BGR", use_column_width=True)

with col2:
    st.subheader("Estado del Recinto")
    
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
    
    # Iconos visuales (estilo panel de control)
    st.write("---")
    st.markdown("### Mapa de Estado")
    status_cols = st.columns(min(len(resultado.boxes), 5))
    for i, box in enumerate(resultado.boxes):
        if i < 5: # Limitamos a 5 iconos para que no se desborde
            label = "🔴" if int(box.cls[0]) == 1 else "🟢"
            status_cols[i].markdown(f"{label} Espacio {i+1}")