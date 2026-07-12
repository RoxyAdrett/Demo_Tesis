import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Q-SOC Parking | UBO",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ESTILOS CSS CORREGIDOS (MÁS COMPACTO Y MODERNO) ---
st.markdown("""
<style>
    /* Ocultar barra lateral y menús innecesarios */
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Fondo general */
    .stApp {
        background-color: #001f3f; /* Azul marino UBO profundo */
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Títulos ajustados para no ser gigantes */
    h1 { font-size: 2.2rem !important; color: #FFFFFF; margin-bottom: 0.5rem; }
    h2 { font-size: 1.5rem !important; color: #00A4E4; }
    h3 { font-size: 1.2rem !important; color: #FFFFFF; }

    /* Tarjetas de métricas - Tamaño equilibrado */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #00A4E4;
        padding: 15px !important;
        border-radius: 8px;
    }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #FFFFFF !important;
    }

    /* Imágenes y bordes */
    .stImage > img {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Footer - Ajustado para que no se vea apartado */
    .footer-container {
        text-align: center;
        padding: 20px;
        margin-top: 20px; /* Reducido de 50px a 20px */
        background-color: rgba(0, 0, 0, 0.2);
        border-top: 1px solid #00A4E4;
    }
    
    /* Spacer para organizar elementos */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt')

modelo = load_model()

# --- HEADER ---
col_head, col_status = st.columns([4, 1])
with col_head:
    st.title("Sistema Inteligente Q-SOC")
    st.markdown("Gestión de Estacionamientos - Facultad de Ingeniería UBO")
with col_status:
    st.markdown("<div style='text-align: right;'>🟢 Sistema En Línea<br>📡 YOLOv11 Activo</div>", unsafe_allow_html=True)

st.markdown("---")

# --- LÓGICA ---
ruta_carpeta = 'ref_images'
imagenes = sorted([f for f in os.listdir(ruta_carpeta) if f.endswith('.jpg')])
selected_img_name = st.selectbox("Seleccione la fuente de entrada:", imagenes)
selected_path = os.path.join(ruta_carpeta, selected_img_name)

img = cv2.imread(selected_path)
resultados = modelo.predict(source=img, imgsz=960, conf=0.25, verbose=False)
resultado = resultados[0]

# Detección
img_bgr = img.copy()
for box in resultado.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    color = (228, 164, 0) if cls == 0 else (50, 50, 255)
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)

# --- CUERPO PRINCIPAL ---
c1, c2 = st.columns([2.5, 1])
with c1:
    st.subheader("🗺️ Visualización")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

with c2:
    st.subheader("📊 Métricas")
    detecciones = [modelo.names[int(box.cls[0])] for box in resultado.boxes]
    df = pd.Series(detecciones).value_counts()
    libres = int(df.get("free_space", 0))
    ocupados = int(df.get("occupied_space", 0))
    
    st.metric("Libres", libres)
    st.metric("Ocupados", ocupados)
    st.metric("Total", libres + ocupados)

# --- GALERÍA ---
st.markdown("---")
st.subheader("📂 Dataset de Referencia")
cols_galeria = st.columns(6)
for i, img_name in enumerate(imagenes[:6]):
    ruta_img = os.path.join(ruta_carpeta, img_name)
    with cols_galeria[i]:
        st.image(Image.open(ruta_img).resize((300, 200)), use_column_width=True)

# --- FOOTER ---
st.markdown("""
<div class="footer-container">
    <p>Sistema Q-SOC | Proyecto de Tesis - Arquitectura y Visión Computacional</p>
    <p style="color: #00A4E4;">© 2026 Desarrollado por <b>Reichell Ardiaca</b> & <b>Sebastian Alvear</b></p>
</div>
""", unsafe_allow_html=True)