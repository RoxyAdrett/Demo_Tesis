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

# --- ESTILOS CSS (IDENTIDAD UBO: AZUL MARINO Y CIAN) ---
st.markdown("""
<style>
    /* Ocultar elementos predeterminados */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Fondo general oscuro con gradiente Azul Marino UBO */
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(135deg, #011528, #002D62);
        color: #FFFFFF;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Textos generales en blanco */
    .stMarkdown p, .stText, h1, h2, h3, h4, h5, h6, label {
        color: #FFFFFF !important;
    }

    /* Resaltados con el Celeste/Cian UBO */
    .ubo-cyan {
        color: #00A4E4 !important;
    }

    /* Estilo de las tarjetas de métricas */
    [data-testid="stMetric"] {
        background-color: rgba(0, 51, 102, 0.4); /* Azul UBO con transparencia */
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 164, 228, 0.3); /* Borde Celeste */
        border-left: 6px solid #00A4E4; /* Borde izquierdo Celeste UBO */
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0px 12px 25px rgba(0, 164, 228, 0.2);
        background-color: rgba(0, 51, 102, 0.6);
    }
    
    /* Valor numérico de la métrica */
    [data-testid="stMetricValue"] {
        color: #00A4E4 !important; /* Celeste UBO */
        font-size: 3rem !important;
        font-weight: 800;
        text-shadow: 0px 0px 10px rgba(0, 164, 228, 0.4);
    }

    /* Contenedores de imágenes */
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0px 10px 25px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Barra de progreso personalizada */
    .stProgress > div > div > div > div {
        background-color: #00A4E4 !important;
    }

    /* Pie de página (Footer) */
    .footer-container {
        text-align: center;
        padding: 30px;
        margin-top: 60px;
        background-color: rgba(0, 15, 35, 0.8);
        border-top: 3px solid #00A4E4;
        border-radius: 12px 12px 0 0;
        box-shadow: 0px -5px 20px rgba(0,0,0,0.3);
    }
    .footer-container p {
        margin: 5px 0;
        font-size: 1rem;
        color: #CCCCCC;
    }
    .footer-logo {
        color: #FFFFFF;
        font-weight: bold;
        font-size: 1.4rem;
        letter-spacing: 2px;
    }
    
    /* Insignias de estado (Badges) */
    .status-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        background-color: rgba(0, 164, 228, 0.15);
        color: #00A4E4;
        border: 1px solid #00A4E4;
        margin-right: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt')

modelo = load_model()

# --- ENCABEZADO (HEADER) INSTITUCIONAL UBO ---
col_logo, col_text = st.columns([1, 5])
with col_logo:
    # Mostramos el logo de la UBO si existe
    if os.path.exists("logo_ubo.png"):
        st.image("logo_ubo.png", use_column_width=True)
    else:
        st.markdown("<h1 style='text-align: center; color: #00A4E4;'>UBO</h1>", unsafe_allow_html=True)

with col_text:
    st.markdown("<h1 style='margin-bottom: 0px;'>Sistema Inteligente Q-SOC</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #00A4E4; margin-top: 0px;'>Gestión de Estacionamientos con Visión Computacional</h4>", unsafe_allow_html=True)
    
    # Badges de estado (Decoración técnica)
    st.markdown("""
        <div class="status-badge">🟢 Sistema En Línea</div>
        <div class="status-badge">📡 Modelo YOLOv11 Activo</div>
    """, unsafe_allow_html=True)

st.markdown("---")

# --- LÓGICA PREVIA ---
ruta_carpeta = 'ref_images'
imagenes = sorted([f for f in os.listdir(ruta_carpeta) if f.endswith('.jpg')])

# --- INTERFAZ PRINCIPAL (DISEÑO A DOS COLUMNAS) ---
col_mapa, col_panel = st.columns([2.2, 1])

with col_panel:
    st.markdown("### <span class='ubo-cyan'>⚙️</span> Panel de Control", unsafe_allow_html=True)
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
    
    # Porcentaje de ocupación para la barra
    pct_ocupacion = (ocupados / total) * 100 if total > 0 else 0

    st.markdown("<br>### <span class='ubo-cyan'>📊</span> Análisis del Recinto", unsafe_allow_html=True)
    st.metric("Libres (Disponibles)", libres)
    st.metric("Ocupados", ocupados)
    
    # Nueva Barra de Ocupación
    st.markdown(f"**Nivel de Ocupación: {pct_ocupacion:.1f}%**")
    st.progress(int(pct_ocupacion))

with col_mapa:
    st.markdown("### <span class='ubo-cyan'>🗺️</span> Mapeo de Detección", unsafe_allow_html=True)
    # Dibujo de cajas limpio
    img_bgr = img.copy()
    for box in resultado.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        # Celeste UBO para libres, Rojo intenso para ocupados
        color = (228, 164, 0) if cls == 0 else (50, 50, 255) # OpenCV usa BGR. (228, 164, 0) es aprox el Celeste UBO
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3) 
    
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

# --- GALERÍA INFERIOR UNIFORME ---
st.markdown("---")
st.markdown("### <span class='ubo-cyan'>📂</span> Historial de Capturas", unsafe_allow_html=True)

TAMANO_GALERIA = (400, 250) 
n_cols = 4
cols_galeria = st.columns(n_cols)

for i, img_name in enumerate(imagenes):
    ruta_img = os.path.join(ruta_carpeta, img_name)
    img_pil = Image.open(ruta_img).resize(TAMANO_GALERIA)
    with cols_galeria[i % n_cols]:
        st.image(img_pil, caption=f"ID Captura: {img_name}", use_column_width=True)

# --- VIDEO DE DEMOSTRACIÓN ---
st.markdown("---")
st.markdown("### <span class='ubo-cyan'>🎥</span> Inferencia en Video", unsafe_allow_html=True)
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
    <span class="footer-logo">Q-SOC PARKING AI</span>
    <p style="margin-top: 10px; font-weight: 500;">Universidad Bernardo O'Higgins - Facultad de Ingeniería</p>
    <p style="color: #00A4E4; font-size: 1.1rem; margin-top: 15px;">© 2026 Desarrollado por <b>Reichell Ardiaca</b> & <b>Sebastian Alvear</b></p>
</div>
""", unsafe_allow_html=True)