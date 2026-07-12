import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
import os
from PIL import Image
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Q-SOC Parking Demo",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS AVANZADOS ---
st.markdown("""
<style>
    /* --- FUENTE Y FONDO GENERAL --- */
    html, body, [class*="st-"] {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    /* Fondo principal con gradiente */
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(120deg, #001f3f, #003366);
        color: #FFFFFF;
    }
    /* Fondo de la barra lateral */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 31, 63, 0.8);
        border-right: 2px solid #F37321; /* Borde naranjo UBO */
    }

    /* --- TÍTULOS Y TEXTOS --- */
    h1, h2, h3 {
        color: #FFFFFF; /* Títulos en blanco para contraste */
    }
    h1 {
        text-align: center;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000000;
    }
    .st-emotion-cache-1g8m2i5, .st-emotion-cache-1kyxreq { /* Labels de métricas y texto general */
        color: #DDDDDD;
    }

    /* --- TARJETAS DE MÉTRICAS --- */
    .st-emotion-cache-1r4qj8v {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
        -webkit-backdrop-filter: blur(4px);
        margin-bottom: 1rem;
    }
    /* Valor de la métrica */
    .st-emotion-cache-1wivapv {
        font-size: 2.5rem !important;
        font-weight: bold;
        color: #F37321; /* Naranjo UBO */
    }

    /* --- CONTENEDORES Y ELEMENTOS --- */
    .st-emotion-cache-z5fcl4 { /* Contenedor principal de elementos */
        border-radius: 15px;
        padding: 2rem;
        background-color: rgba(0, 0, 0, 0.2);
    }
    .stImage > img { /* Estilo para imágenes */
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.5);
    }
    .stVideo > video { /* Estilo para video */
        border-radius: 10px;
    }
    /* Efecto hover en galería */
    div[data-testid*="stImage"][style*="transform: scale(1);"] {
        transition: transform 0.3s ease-in-out;
    }
    div[data-testid*="stImage"][style*="transform: scale(1);"]:hover {
        transform: scale(1.05);
    }

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO('best.pt')

# --- CARGA DE MODELO ---
modelo = load_model()

# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    # Logo UBO (reemplaza 'logo_ubo.png' si tienes el archivo localmente)
    # Por ahora, usamos un placeholder de texto. Si tienes el logo, descomenta la línea de st.image
    st.markdown("<h2 style='text-align: center; color: #F37321;'>UBO</h2>", unsafe_allow_html=True)
    # st.image("ruta/a/tu/logo_ubo.png", use_column_width=True)
    st.header("Panel de Control")

    # Selector de imágenes
    ruta_carpeta = 'ref_images'
    imagenes = sorted([f for f in os.listdir(ruta_carpeta) if f.endswith('.jpg')])
    selected_img_name = st.selectbox("Selecciona imagen a analizar:", imagenes, label_visibility="collapsed")
    
    st.image(Image.open(os.path.join(ruta_carpeta, selected_img_name)), use_column_width=True)

    # Copyright
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; font-size: 0.9rem;'>"
        "© 2024 - Sistema Q-SOC<br>"
        "Desarrollado por:<br>"
        "<b>Reichell Ardiaca</b> & <b>Sebastian Alvear</b>"
        "</div>",
        unsafe_allow_html=True
    )

# --- TÍTULO PRINCIPAL ---
st.title("Sistema Inteligente de Estacionamientos (Q-SOC)")
st.markdown("---")

# --- LÓGICA DE PREDICCIÓN ---
ruta_carpeta = 'ref_images'
selected_path = os.path.join(ruta_carpeta, selected_img_name)
img = cv2.imread(selected_path)
resultados = modelo.predict(source=img, imgsz=960, conf=0.25, verbose=False)
resultado = resultados[0]

# Dibujo de bounding boxes en la imagen
img_bgr = img.copy()
for box in resultado.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls = int(box.cls[0])
    color = (0, 255, 0) if cls == 0 else (50, 50, 255) # Verde para libre, Rojo para ocupado
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)

# --- INTERFAZ PRINCIPAL ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Visualización del Mapa")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

with col2:
    st.subheader("Estado del Estacionamiento")
    detecciones = [modelo.names[int(box.cls[0])] for box in resultado.boxes]
    df = pd.Series(detecciones).value_counts()
    
    libres = int(df.get("free_space", 0))
    ocupados = int(df.get("occupied_space", 0))
    total = libres + ocupados

    st.metric("✅ Espacios Libres", libres)
    st.metric("❌ Espacios Ocupados", ocupados)
    st.metric("🅿️ Total de Espacios", total)

# --- GALERÍA INFERIOR ---
with st.container():
    st.markdown("---")
    st.subheader("Dataset de Referencia")
    
    n_cols = 4 # Número de imágenes por fila
    cols_galeria = st.columns(n_cols)
    for i, img_name in enumerate(imagenes):
        with cols_galeria[i % n_cols]:
            st.image(os.path.join(ruta_carpeta, img_name), caption=img_name, use_column_width=True)

# --- VIDEO ---
with st.container():
    st.markdown("---")
    st.subheader("Demostración de Procesamiento en Video")
    
    video_path = 'demo_video.mp4'
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.warning(f"El video '{video_path}' no se encuentra en la raíz del proyecto.")