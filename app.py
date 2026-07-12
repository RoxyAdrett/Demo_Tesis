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

# --- ESTILOS CSS (PANEL CRISTAL OSCURO Y CORRECCIÓN DE LOGO) ---
st.markdown("""
<style>
    /* Ocultar barra lateral y menús predeterminados */
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Fondo general de la página (Gradiente UBO que se verá en los bordes) */
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(135deg, #011528, #002D62);
        background-attachment: fixed;
        color: #FFFFFF;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* --- EL NUEVO CONTENEDOR SEPARADOR (Efecto Glassmorphism Oscuro) --- */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 95% !important;
        margin-top: 3vh; /* Margen superior para ver el fondo */
        margin-bottom: 3vh; /* Margen inferior para ver el fondo */
        background-color: rgba(2, 12, 27, 0.75) !important; /* Panel oscuro translúcido */
        backdrop-filter: blur(12px); /* Efecto cristal borroso al fondo */
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(0, 164, 228, 0.25); /* Borde sutil Celeste UBO */
        box-shadow: 0px 25px 50px rgba(0,0,0,0.7); /* Sombra profunda para despegarlo del fondo */
    }

    /* Textos generales en blanco */
    .stMarkdown p, .stText, h1, h2, h3, h4, h5, h6, label {
        color: #FFFFFF !important;
    }

    /* Resaltados con el Celeste/Cian UBO */
    .ubo-cyan {
        color: #00A4E4 !important;
    }

    /* Alineación vertical del Encabezado (Centra el Logo y el Texto) */
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }

    /* --- CORRECCIÓN DEL LOGO --- */
    /* Le quitamos el borde y sombra específicamente a la primera imagen (el logo) */
    div[data-testid="column"]:nth-child(1) div[data-testid="stImage"] img {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        border-radius: 0px;
        max-height: 100px; /* Controla el tamaño máximo del logo */
    }

    /* --- TAMAÑO DE LAS IMÁGENES PRINCIPALES Y ANIMACIÓN --- */
    /* Esto solo afectará a las imágenes de las cámaras y la galería */
    div[data-testid="column"]:nth-child(n+2) div[data-testid="stImage"] img,
    div[data-testid="stImage"]:nth-child(n+2) img {
        border-radius: 8px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.4);
        border: 1px solid rgba(0, 164, 228, 0.3);
        max-height: 60vh;
        object-fit: contain;
        margin: 0 auto;
        transition: transform 0.3s ease-in-out;
    }
    
    div[data-testid="column"]:nth-child(n+2) div[data-testid="stImage"] img:hover {
        transform: scale(1.02);
    }

    /* Estilo de las tarjetas de métricas */
    [data-testid="stMetric"] {
        background-color: rgba(0, 30, 60, 0.6); /* Más oscuras para contrastar con el panel */
        border: 1px solid rgba(0, 164, 228, 0.3);
        border-left: 6px solid #00A4E4; 
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.5);
        margin-bottom: 1rem;
        transition: transform 0.2s ease-in-out;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        background-color: rgba(0, 40, 80, 0.8);
    }
    
    /* Valor numérico de la métrica */
    [data-testid="stMetricValue"] {
        color: #00A4E4 !important; 
        font-size: 2.5rem !important;
        font-weight: 800;
        text-shadow: 0px 0px 10px rgba(0, 164, 228, 0.4);
    }

    /* Barra de progreso personalizada */
    .stProgress > div > div > div > div {
        background-color: #00A4E4 !important;
    }

    /* Pie de página (Footer) integrado al panel */
    .footer-container {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        background-color: transparent;
        border-top: 1px solid rgba(0, 164, 228, 0.3);
    }
    .footer-container p {
        margin: 5px 0;
        font-size: 0.95rem;
        color: #AAAAAA;
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
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt')

modelo = load_model()

# --- ENCABEZADO (HEADER) INSTITUCIONAL UBO ---
col_logo, col_text = st.columns([1.5, 6]) # Ajustado el ancho para que el logo respire
with col_logo:
    if os.path.exists("logo_ubo.png"):
        st.image("logo_ubo.png")
    else:
        st.markdown("<h1 style='text-align: center; color: #00A4E4; font-size: 3rem;'>UBO</h1>", unsafe_allow_html=True)

with col_text:
    st.markdown("<h1 style='margin-bottom: 0px;'>Predicción de ocupación y gestión de estacionamientos</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #00A4E4; margin-top: 5px; margin-bottom: 15px;'>Gestión de Estacionamientos con Visión Computacional</h4>", unsafe_allow_html=True)
    
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
    st.markdown("<h3><span class='ubo-cyan'>⚙️</span> Panel de Control</h3>", unsafe_allow_html=True)
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

    st.markdown("<br><h3><span class='ubo-cyan'>📊</span> Análisis del Recinto</h3>", unsafe_allow_html=True)
    st.metric("Libres (Disponibles)", libres)
    st.metric("Ocupados", ocupados)
    
    # Barra de Ocupación
    st.markdown(f"**Nivel de Ocupación: {pct_ocupacion:.1f}%**")
    st.progress(int(pct_ocupacion))

with col_mapa:
    st.markdown("<h3><span class='ubo-cyan'>🗺️</span> Mapeo de Detección</h3>", unsafe_allow_html=True)
    
    img_bgr = img.copy()
    for box in resultado.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        color = (228, 164, 0) if cls == 0 else (50, 50, 255) 
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3) 
    
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

# --- GALERÍA INFERIOR ---
st.markdown("---")
st.markdown("<h3><span class='ubo-cyan'>📂</span> Historial de Capturas</h3>", unsafe_allow_html=True)

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
st.markdown("<h3><span class='ubo-cyan'>🎥</span> Inferencia en Video Continuo</h3>", unsafe_allow_html=True)
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
    <p style="color: #00A4E4; margin-top: 10px;">© 2026 Desarrollado por <b>Reichell Ardiaca</b> & <b>Sebastian Alvear</b></p>
</div>
""", unsafe_allow_html=True)