import streamlit as st
import cv2
from ultralytics import YOLO

# 1. Tu modelo detecta las cajas
results = model(frame)

# 2. Crea un contenedor para tu "interfaz" personalizada
st.subheader("Mapa de Estacionamientos en Tiempo Real")
cols = st.columns(len(results[0].boxes)) # Crea una columna por cada espacio detectado

for i, box in enumerate(results[0].boxes):
    # box.cls da 0 (free) o 1 (occupied)
    is_occupied = int(box.cls) == 1 
    
    # Aquí dibujas tu icono personalizado
    with cols[i]:
        if is_occupied:
            st.markdown("🔴 **OCUPADO**") # O usa st.image("icono_auto_rojo.png")
        else:
            st.markdown("🟢 **LIBRE**")   # O usa st.image("icono_auto_verde.png")