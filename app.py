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

# --- ESTILOS CSS (FONDO AZUL ORIGINAL + ACENTOS BLANCOS + RESPONSIVE) ---
st.markdown("""
<style>
    /* Ocultar barra lateral y menús predeterminados */
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* --- FONDO GENERAL (Gradiente Azul Marino UBO en toda la pantalla) --- */
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(135deg, #011528, #002D62) !important;
        background-attachment: fixed;
        color: #FFFFFF;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* Reducir márgenes internos de Streamlit */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 95% !important;
    }

    /* --- RESPONSIVE DESIGN (Ajustes para celular) --- */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem !important;
            max-width: 100% !important;
        }
        h1 { font-size: 1.8rem !important; line-height: 1.2 !important; }
        h3 { font-size: 1.3rem !important; }
        .footer-container p { font-size: 0.85rem !important; }
    }

    /* Textos generales en blanco */
    .stMarkdown p, .stText, h1, h2, h3, h4, h5, h6, label {
        color: #FFFFFF !important;
    }

    /* Clase para iconos/detalles que antes eran celestes y ahora son blancos */
    .ubo-white {
        color: #FFFFFF !important;
    }

    /* --- CORRECCIÓN DEL LOGO --- */
    div[data-testid="column"]:nth-child(1) div[data-testid="stImage"] img {
        border: none !important;
        background-color: #FFFFFF !important;
        border-radius: 12px;
        padding: 8px;
        box-shadow: 0px 4px 15px rgba(255,255,255,0.1) !important;
        max-height: 100px;
        object-fit: contain;
    }

    /* --- TAMAÑO DE LAS IMÁGENES PRINCIPALES Y ANIMACIÓN --- */
    div[data-testid="column"]:nth-child(n+2) div[data-testid="stImage"] img,
    div[data-testid="stImage"]:nth-child(n+2) img {
        border-radius: 8px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.3); /* Borde blanco sutil */
        max-height: 60vh;
        object-fit: contain;
        margin: 0 auto;
        transition: transform 0.3s ease-in-out;
    }
    
    div[data-testid="column"]:nth-child(n+2) div[data-testid="stImage"] img:hover {
        transform: scale(1.02);
    }

    /* --- TARJETAS DE MÉTRICAS (Estilo Cristal con acentos Blancos) --- */
    [data-testid="stMetric"] {
        background-color: rgba(0, 30, 60, 0.5); /* Azul profundo translúcido */
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-left: 6px solid #FFFFFF; /* Borde izquierdo BLANCO */
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
        transition: transform 0.2s ease-in-out;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        background-color: rgba(0, 40, 80, 0.7);
    }
    
    [data-testid="stMetricValue"] {
        color: #FFFFFF !important; /* Valor en BLANCO */
        font-size: 2.8rem !important;
        font-weight: 800;
        text-shadow: 0px 0px 10px rgba(255, 255, 255, 0.3);
    }

    /* Barra de progreso personalizada (Blanca) */
    .stProgress > div > div > div > div {
        background-color: #FFFFFF !important;
    }

    /* Pie de página (Footer) */
    .footer-container {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        background-color: transparent;
        border-top: 1px solid rgba(255, 255, 255, 0.2);
    }
    .footer-container p {
        margin: 5px 0;
        font-size: 0.95rem;
        color: #CCCCCC;
    }
    
    /* Insignias de estado (Badges Blancos) */
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        background-color: rgba(255, 255, 255, 0.1); /* Fondo blanco semitransparente */
        color: #FFFFFF;
        border: 1px solid #FFFFFF;
        margin-right: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt')

modelo = load_model()

# --- ENCABEZADO (HEADER) INSTITUCIONAL UBO ---
col_logo, col_text = st.columns([1.5, 6]) 
with col_logo:
    if os.path.exists("logo_ubo.png"):
        st.image("logo_ubo.png")
    else:
        st.markdown("<h1 style='text-align: center; color: #FFFFFF; font-size: 3rem;'>UBO</h1>", unsafe_allow_html=True)

with col_text:
    st.markdown("<h1 style='margin-bottom: 0px;'>Predicción de ocupación y gestión de estacionamientos</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #DDDDDD; margin-top: 5px; margin-bottom: 15px;'>Gestión de Estacionamientos con Visión Computacional</h4>", unsafe_allow_html=True)
    
    # Badges de estado 
    st.markdown("""
        <div class="status-badge">🟢 Sistema En Línea</div>
        <div class="status-badge">📡 Modelo YOLOv11 Activo</div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- LÓGICA PREVIA ---
ruta_carpeta = 'ref_images'
imagenes = sorted([f for f in os.listdir(ruta_carpeta) if f.endswith('.jpg')])

# --- INTERFAZ PRINCIPAL (DISEÑO A DOS COLUMNAS) ---
col_mapa, col_panel = st.columns([2.5, 1.2]) 

with col_panel:
    st.markdown("<h3><span class='ubo-white'>⚙️</span> Panel de Control</h3>", unsafe_allow_html=True)
    selected_img_name = st.selectbox("Seleccione la fuente de entrada:", imagenes)
    
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
    
    # Porcentaje de ocupación
    pct_ocupacion = (ocupados / total) * 100 if total > 0 else 0

    st.markdown("<br><h3><span class='ubo-white'>📊</span> Análisis del Recinto</h3>", unsafe_allow_html=True)
    st.metric("Libres (Disponibles)", libres)
    st.metric("Ocupados", ocupados)
    
    # Barra de Ocupación
    st.markdown(f"**Nivel de Ocupación: {pct_ocupacion:.1f}%**")
    st.progress(int(pct_ocupacion))

with col_mapa:
    st.markdown("<h3><span class='ubo-white'>🗺️</span> Mapeo de Detección</h3>", unsafe_allow_html=True)
    
    img_bgr = img.copy()
    for box in resultado.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        # Blanco (255,255,255) para libres, Rojo intenso (50,50,255) para ocupados
        color = (255, 255, 255) if cls == 0 else (50, 50, 255) 
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3) 
    
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

# --- GALERÍA INFERIOR ---
st.markdown("---")
st.markdown("<h3><span class='ubo-white'>📂</span> Historial de Capturas</h3>", unsafe_allow_html=True)

TAMANO_GALERIA = (300, 200) 
n_cols = 5
cols_galeria = st.columns(n_cols)

for i, img_name in enumerate(imagenes[:5]): 
    ruta_img = os.path.join(ruta_carpeta, img_name)
    img_pil = Image.open(ruta_img).resize(TAMANO_GALERIA)
    with cols_galeria[i]:
        st.image(img_pil, caption=f"{img_name}", use_container_width=True)

# --- VIDEO DE DEMOSTRACIÓN ---
st.markdown("---")
st.markdown("<h3><span class='ubo-white'>🎥</span> Inferencia en Video Continuo</h3>", unsafe_allow_html=True)
video_path = 'demo_video.mp4'
if os.path.exists(video_path):
    v_col1, v_col2, v_col3 = st.columns([1, 3, 1])
    with v_col2:
        st.video(video_path)
else:
    st.warning(f"El video '{video_path}' no se encuentra en el repositorio.")

# --- FOOTER INSTITUCIONAL ---
st.markdown("""
<div class="footer-container">
    <p style="font-weight: 600; color: #FFFFFF; font-size: 1.1rem; letter-spacing: 1px;">Predicción de ocupación y gestión de estacionamientos</p>
    <p>Universidad Bernardo O'Higgins - Facultad de Ingeniería</p>
    <p style="color: #FFFFFF; margin-top: 10px;">© 2026 Desarrollado por <b>Reichell Ardiaca</b> & <b>Sebastian Alvear</b></p>
</div>
""", unsafe_allow_html=True)