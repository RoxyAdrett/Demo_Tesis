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

# --- ESTILOS CSS (IDENTIDAD UBO + PROPORCIONES CORREGIDAS) ---
st.markdown("""
<style>
    /* Ocultar barra lateral y menús predeterminados */
    [data-testid="collapsedControl"] { display: none; }
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

    /* --- CORRECCIÓN: TAMAÑO DE LA IMAGEN PRINCIPAL --- */
    div[data-testid="stImage"] img {
        border-radius: 8px;
        box-shadow: 0px 8px 20px rgba(0,0,0,0.4);
        border: 1px solid rgba(0, 164, 228, 0.3);
        max-height: 60vh; /* Límite de altura a la proporción de la pantalla */
        object-fit: contain; /* Evita que se deforme */
        margin: 0 auto; /* Centra la imagen */
    }

    /* Estilo de las tarjetas de métricas */
    [data-testid="stMetric"] {
        background-color: rgba(0, 51, 102, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 164, 228, 0.3);
        border-left: 6px solid #00A4E4; 
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
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

    /* Pie de página (Footer) - Acoplado correctamente */
    .footer-container {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        background-color: rgba(0, 15, 35, 0.8);
        border-top: 3px solid #00A4E4;
        border-radius: 12px 12px 0 0;
    }
    .footer-container p {
        margin: 5px 0;
        font-size: 0.95rem;
        color: #CCCCCC;
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
col_logo, col_text = st.columns([1, 6])
with col_logo:
    if os.path.exists("logo_ubo.png"):
        st.image("logo_ubo.png")
    else:
        st.markdown("<h1 style='text-align: center; color: #00A4E4; font-size: 3rem;'>UBO</h1>", unsafe_allow_html=True)

with col_text:
    st.markdown("<h1 style='margin-bottom: 0px;'>Sistema Inteligente Q-SOC</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #00A4E4; margin-top: 0px;'>Gestión de Estacionamientos con Visión Computacional</h4>", unsafe_allow_html=True)
    
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
col_mapa, col_panel = st.columns([2.5, 1.2]) # Ajusté la proporción para que el mapa no se estire tanto

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
    
    # Porcentaje de ocupación
    pct_ocupacion = (ocupados / total) * 100 if total > 0 else 0

    st.markdown("<br>### <span class='ubo-cyan'>📊</span> Análisis del Recinto", unsafe_allow_html=True)
    st.metric("Libres (Disponibles)", libres)
    st.metric("Ocupados", ocupados)
    
    # Barra de Ocupación restaurada
    st.markdown(f"**Nivel de Ocupación: {pct_ocupacion:.1f}%**")
    st.progress(int(pct_ocupacion))

with col_mapa:
    st.markdown("### <span class='ubo-cyan'>🗺️</span> Mapeo de Detección", unsafe_allow_html=True)
    
    img_bgr = img.copy()
    for box in resultado.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        color = (228, 164, 0) if cls == 0 else (50, 50, 255) # Celeste UBO libre, Rojo ocupado
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3) 
    
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

# --- GALERÍA INFERIOR ---
st.markdown("---")
st.markdown("### <span class='ubo-cyan'>📂</span> Historial de Capturas", unsafe_allow_html=True)

TAMANO_GALERIA = (300, 200) 
n_cols = 5
cols_galeria = st.columns(n_cols)

for i, img_name in enumerate(imagenes[:5]): # Limitado a 5 para no saturar
    ruta_img = os.path.join(ruta_carpeta, img_name)
    img_pil = Image.open(ruta_img).resize(TAMANO_GALERIA)
    with cols_galeria[i]:
        st.image(img_pil, caption=f"{img_name}", use_container_width=True)

# --- VIDEO DE DEMOSTRACIÓN ---
st.markdown("---")
st.markdown("### <span class='ubo-cyan'>🎥</span> Inferencia en Video Continuo", unsafe_allow_html=True)
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
    <p style="font-weight: 600; color: #FFFFFF; font-size: 1.1rem; letter-spacing: 1px;">Q-SOC PARKING AI</p>
    <p>Universidad Bernardo O'Higgins - Facultad de Ingeniería</p>
    <p style="color: #00A4E4; margin-top: 10px;">© 2026 Desarrollado por <b>Reichell Ardiaca</b> & <b>Sebastian Alvear</b></p>
</div>
""", unsafe_allow_html=True)