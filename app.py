import streamlit as st
import cv2
from ultralytics import YOLO

# Configuración de página
st.set_page_config(layout="wide", page_title="Demo Parking AI")
st.title("Sistema Inteligente de Estacionamientos (Q-SOC)")

# Carga del modelo
model = YOLO('best.pt')

# Layout: Columnas para organizar
col_video, col_info = st.columns([2, 1])

with col_video:
    st.subheader("Monitoreo en Tiempo Real")
    video_placeholder = st.empty()

with col_info:
    st.subheader("Estado del Recinto")
    stats_placeholder = st.empty()

# Procesamiento
cap = cv2.VideoCapture('demo_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Inferencia
    results = model(frame)
    res_plotted = results[0].plot()
    
    # Cálculos de lógica (conteo)
    freespace = sum(1 for box in results[0].boxes.cls if box == 0) # Asumiendo 0=free
    occupied = sum(1 for box in results[0].boxes.cls if box == 1) # Asumiendo 1=occ
    
    # Actualizar UI
    video_placeholder.image(res_plotted, channels="BGR")
    stats_placeholder.metric("Espacios Libres", freespace)
    stats_placeholder.metric("Espacios Ocupados", occupied)

# Galería de referencia abajo
st.divider()
st.subheader("Dataset de Referencia")
cols_ref = st.columns(4)
# Aquí puedes cargar tus imágenes de referencia
for i, img_path in enumerate(['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']):
    cols_ref[i].image(img_path, caption=f"Ref {i+1}")