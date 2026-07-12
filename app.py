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
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ESTILOS CSS AVANZADOS Y MODERNOS ---
st.markdown("""
<style>
    /* Ocultar elementos predeterminados de Streamlit para un look más limpio */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Fondo general */
    .stApp {
        background-color: #F4F7F6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Títulos principales */
    h1 {
        color: #002D62; /* Azul Institucional */
        font-weight: 800;
        margin-bottom: 0rem;
    }
    h2, h3 {
        color: #004080;
        font-weight: 600;
    }

    /* Estilo de las tarjetas de métricas */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border-left: 6px solid #F37321; /* Naranjo UBO */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease-in-out;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
    }
    [data-testid="stMetricValue"] {
        color: #002D62;
        font-size: 2.5rem;
        font-weight: bold;
    }

    /* Contenedores de imágenes */
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
        border: 2px solid #EAEAEA;
    }

    /* Pie de página (Footer) */
    .footer-container {
        text-align: center;
        padding: 25px;
        margin-top: 50px;
        background-color: #002D62;
        color: #FFFFFF;
        border-radius: 10px 10px 0 0;
        box-shadow: 0px -4px 10px rgba(0,0,0,0.1);
    }
    .footer-container p {
        margin: 5px 0;
        font-size: 1rem;
    }
    .footer-logo {
        color: #F37321;
        font-weight: bold;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt')

modelo = load_model()

# --- ENCABEZADO (HEADER) INSTITUCIONAL ---
col_logo, col_text = st.columns([1, 6])
with col_logo:
    # Puedes cambiar el emoji por tu st.image("logo_ubo.png") si lo tienes
    st.markdown("<h1 style='font-size: 4rem; text-align: center; color: #F37321;'>🎓</h1>", unsafe_allow_html=True)
with col_text:
    st.title("Sistema Inteligente de Estacionamientos (Q-SOC)")
    st.markdown("**Proyecto de Visión Computacional - Universidad Bernardo O'Higgins**")

st.markdown("---")

# --- LÓGICA PREVIA ---
ruta_carpeta = 'ref_images'
imagenes = sorted([f for f in os.listdir(ruta_carpeta) if f.endswith('.jpg')])

# --- INTERFAZ PRINCIPAL (DISEÑO A DOS COLUMNAS) ---
col_mapa, col_panel = st.columns([2.2, 1])

with col_panel:
    st.markdown("### 🎛️ Panel de Control")
    selected_img_name = st.selectbox("Seleccione la fuente de video/imagen:", imagenes)
    
    # Procesamiento
    selected_path = os.path.join(ruta_carpeta, selected_img_name)
    img = cv2.imread(selected_path)
    resultados = modelo.predict(source=img, imgsz=960, conf=0.25, verbose=False)
    resultado = resultados[0]

    # Cálculos
    detecciones = [modelo.names[int(box.cls[0])] for box in resultado.boxes]
    df = pd.Series(detecciones).value_counts()
    libres = int(df.get("free_space", 0))
    ocupados = int(df.get("occupied_space", 0))
    total = libres + ocupados

    st.markdown("<br>### 📊 Análisis en Tiempo Real", unsafe_allow_html=True)
    st.metric("✅ Espacios Libres", libres)
    st.metric("❌ Espacios Ocupados", ocupados)
    st.metric("🅿️ Total Capacidad", total)

with col_mapa:
    st.markdown("### 🗺️ Visualización de Ocupación")
    # Dibujo de cajas limpio (sin texto)
    img_bgr = img.copy()
    for box in resultado.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        color = (0, 255, 0) if cls == 0 else (50, 50, 255) # Verde libre, Rojo ocupado
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 4) # Grosor 4 para que se vea mejor
    
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

# --- GALERÍA INFERIOR UNIFORME ---
st.markdown("---")
st.markdown("### 📂 Dataset de Referencia")

# Redimensionamiento estricto para que la galería no se vea deforme
TAMANO_GALERIA = (400, 250) 
n_cols = 4
cols_galeria = st.columns(n_cols)

for i, img_name in enumerate(imagenes):
    ruta_img = os.path.join(ruta_carpeta, img_name)
    # Abrimos, redimensionamos y mostramos para asegurar simetría total
    img_pil = Image.open(ruta_img).resize(TAMANO_GALERIA)
    with cols_galeria[i % n_cols]:
        st.image(img_pil, caption=f"Muestra: {img_name}", use_column_width=True)

# --- VIDEO DE DEMOSTRACIÓN ---
st.markdown("---")
st.markdown("### 🎥 Demostración de Procesamiento Continuo")
video_path = 'demo_video.mp4'
if os.path.exists(video_path):
    # Centrar el video usando columnas
    v_col1, v_col2, v_col3 = st.columns([1, 3, 1])
    with v_col2:
        st.video(video_path)
else:
    st.warning(f"El video '{video_path}' no se encuentra en el repositorio.")

# --- FOOTER (DERECHOS DE AUTOR) ---
st.markdown("""
<div class="footer-container">
    <span class="footer-logo">Sistema Q-SOC</span>
    <p>Proyecto de Tesis - Arquitectura y Visión Computacional</p>
    <p>© 2026 Desarrollado por <b>Reichell Ardiaca</b> & <b>Sebastian Alvear</b></p>
</div>
""", unsafe_allow_html=True)