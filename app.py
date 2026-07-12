import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="IA Parking | UBO",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ESTILOS CSS ---
st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu, header, footer {visibility: hidden;}

    [data-testid="stAppViewContainer"] {
        background-color: #011528;
        background-image: 
            radial-gradient(rgba(255, 255, 255, 0.15) 2px, transparent 2px),
            linear-gradient(135deg, #011528, #002D62) !important;
        background-size: 40px 40px, 100% 100%;
        background-attachment: fixed;
        color: #FFFFFF;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    .block-container {
        padding: 2.5rem;
        background: rgba(0, 18, 40, 0.75);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border-left: 3px solid #FFFFFF; 
        border-right: 3px solid #FFFFFF; 
    }

    /* Animación solo para elementos pequeños */
    .stMetric, .stImage img { transition: transform 0.3s ease; }
    .stMetric:hover { transform: scale(1.02); }

    /* Forzar tamaño real en la imagen del mapa (Sin animación ni restricciones) */
    div[data-testid="stImage"] img {
        width: 100% !important;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Logo sin animación ni bordes raros */
    .logo-container img {
        background: transparent !important;
        box-shadow: none !important;
        max-height: 100px;
    }

    .status-badge {
        display: inline-block; padding: 5px 12px; border-radius: 20px;
        background: rgba(255, 255, 255, 0.1); border: 1px solid #FFFFFF;
        margin-right: 10px; font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt')

modelo = load_model()

# --- HEADER ---
col_logo, col_text = st.columns([1.5, 6])
with col_logo:
    if os.path.exists("logo_ubo.png"):
        st.image("logo_ubo.png")
with col_text:
    st.markdown("<h1>Predicción de ocupación y gestión de estacionamientos</h1>", unsafe_allow_html=True)
    st.markdown("<h4>Gestión de Estacionamientos con Visión Computacional</h4>", unsafe_allow_html=True)
    st.markdown('<div class="status-badge">🟢 Sistema En Línea</div><div class="status-badge">📡 Modelo YOLOv11 Activo</div>', unsafe_allow_html=True)

st.markdown("---")

# --- LÓGICA ---
ruta_carpeta = 'ref_images'
imagenes = sorted([f for f in os.listdir(ruta_carpeta) if f.endswith('.jpg')])

col1, col2 = st.columns([2.5, 1.2])

with col1:
    st.subheader("🗺️ Mapeo de Detección")
    selected_img = st.selectbox("Seleccione la fuente de entrada:", imagenes)
    img_path = os.path.join(ruta_carpeta, selected_img)
    img = cv2.imread(img_path)
    res = modelo.predict(source=img, imgsz=960, conf=0.25, verbose=False)[0]
    
    img_bgr = img.copy()
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (228, 164, 0) if int(box.cls[0]) == 0 else (50, 50, 255)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

with col2:
    st.subheader("⚙️ Panel de Control")
    detecciones = [modelo.names[int(box.cls[0])] for box in res.boxes]
    df = pd.Series(detecciones).value_counts()
    libres = int(df.get("free_space", 0))
    ocupados = int(df.get("occupied_space", 0))
    total = libres + ocupados
    
    st.metric("Libres (Disponibles)", libres)
    st.metric("Ocupados", ocupados)
    st.progress((ocupados / total) if total > 0 else 0)

# --- GALERÍA ---
st.markdown("---")
st.subheader("📂 Historial de Capturas")
cols = st.columns(5)
for i, img_name in enumerate(imagenes[:5]):
    with cols[i]:
        st.image(Image.open(os.path.join(ruta_carpeta, img_name)).resize((300, 200)), use_container_width=True)

# --- FOOTER ---
st.markdown("""
<div style="text-align: center; margin-top: 30px; border-top: 1px solid #FFF; padding: 20px;">
    <p>© 2026 Desarrollado por <b>Reichell Ardiaca</b> & <b>Sebastian Alvear</b></p>
</div>
""", unsafe_allow_html=True)