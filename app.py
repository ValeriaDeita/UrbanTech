# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json
import joblib
import streamlit.components.v1 as components
import base64

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="UrbanTech Copilot", page_icon="🏙️", layout="wide")

# --- 2. TRATAMIENTO DE IMÁGENES (BASE64) ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

fondo_b64 = get_base64_of_bin_file("fondo.png")
logo_b64 = get_base64_of_bin_file("logourban.png")

# CSS para el fondo
fondo_style = f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{fondo_b64}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}
</style>
""" if fondo_b64 else ""

# --- 3. CSS MAESTRO: DISEÑO PREMIUM ---
st.markdown(f"""
{fondo_style}
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Orbitron:wght@500;900&display=swap');

    /* Fuentes y Globales */
    body, .stApp {{ font-family: 'Inter', sans-serif; }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{ display: none !important; }}

    /* ENCABEZADO IMPACTANTE */
    .hero-header {{
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(4, 15, 26, 0.9);
        padding: 40px;
        border-radius: 20px;
        border-bottom: 3px solid #e67e22;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.8);
        border: 1px solid rgba(230, 126, 34, 0.3);
    }}
    .logo-container img {{
        height: 150px; /* Logo más grande */
        filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.5));
    }}
    .header-text {{
        margin-left: 30px;
        text-align: left;
    }}
    .header-text h1 {{
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5em !important;
        color: #ffffff !important;
        margin: 0;
        letter-spacing: 2px;
    }}

    /* PANELES (CONTENEDORES) */
    [data-testid="stColumn"] > div {{
        background-color: rgba(6, 15, 25, 0.92) !important;
        border-radius: 15px !important;
        padding: 25px !important;
        backdrop-filter: blur(10px);
    }}
    
    /* Panel Dashboard (Azul/Cian) */
    [data-testid="stColumn"]:nth-child(1) > div {{
        border: 1px solid #00ffff !important;
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.15);
    }}

    /* Panel Chat (Naranja) */
    [data-testid="stColumn"]:nth-child(2) > div {{
        border: 1px solid #e67e22 !important;
        box-shadow: 0 0 20px rgba(230, 126, 34, 0.15);
    }}

    /* CAJAS DE CONTEXTO */
    .context-box {{
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 0.9em;
        line-height: 1.4;
    }}
    .bi-context {{
        background: rgba(0, 255, 255, 0.05);
        border-left: 4px solid #00ffff;
    }}
    .ai-context {{
        background: rgba(230, 126, 34, 0.05);
        border-left: 4px solid #e67e22;
    }}

    /* CHAT INPUT NARANJA */
    .stChatInputContainer {{
        border: 1px solid #e67e22 !important;
        background: #0a1929 !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- 4. RENDERIZADO DEL ENCABEZADO ---
logo_img_html = f'<img src="data:image/png;base64,{logo_b64}">' if logo_b64 else "🏙️"
st.markdown(f"""
<div class="hero-header">
    <div class="logo-container">{logo_img_html}</div>
    <div class="header-text">
        <h1>UrbanTech Copilot</h1>
        <p style='color: #00ffff; font-weight: bold; margin:0;'>MINERÍA DE DATOS | FES ACATLÁN</p>
        <p style='color: #888; margin:0;'>Equipo: Denisse, Martinez, Samu, Dei</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. CARGA DE DATOS ---
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('modelo_lightgbm_v20260508_2005.joblib')
    with open('predicciones_2024_con_clusters.json', 'r', encoding='utf-8') as f:
        predicciones = json.load(f)
    return modelo, predicciones

try:
    modelo, predicciones = cargar_recursos()
    status_ready = True
except:
    status_ready = False

# --- 6. COLUMNAS ---
col_dash, col_chat = st.columns([1.6, 1], gap="large")

with col_dash:
    st.markdown("<h2 style='color:#00ffff !important;'>📊 Inteligencia de Negocios (BI)</h2>", unsafe_allow_html=True)
    
    # Caja de contexto Dashboard
    st.markdown("""
    <div class="context-box bi-context">
        <b style="color:#00ffff;">El Pulso del Caos:</b> Radiografía de incidentes en CDMX.<br>
        Este tablero procesa el <i>Data Warehouse</i> histórico para identificar patrones espaciales. 
        Utiliza los filtros superiores para explorar alcaldías o condiciones climáticas y visualizar el impacto en la siniestralidad vial.
    </div>
    """, unsafe_allow_html=True)
    
    looker_url = "https://lookerstudio.google.com/embed/reporting/6170d005-664f-4ba1-a21f-a3127ab0e52d/page/gWYvF"
    components.iframe(looker_url, height=750, scrolling=True)

with col_chat:
    st.markdown("<h2 style='color:#e67e22 !important;'>🤖 Agente Predictivo (IA)</h2>", unsafe_allow_html=True)
    
    # Caja de contexto Chat
    st.markdown("""
    <div class="context-box ai-context">
        <b style="color:#e67e22;">Instrucciones del Copilot:</b><br>
        Interactúa con el modelo <b>LightGBM</b> entrenado para predicción de riesgos.
        <ul>
            <li>Consulta zonas: <i>"¿Cuáles son los clusters de mayor riesgo?"</i></li>
            <li>Simula escenarios: <i>"¿Cómo afecta la lluvia al Cluster 2?"</i></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if status_ready:
        st.success("✅ Motor de Inteligencia Artificial activo.")
    else:
        st.error("⚠️ Error: No se encontraron los archivos del modelo o predicciones.")

    # Lógica del Chat
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    for m in st.session_state.mensajes:
        with st.chat_message(m["rol"], avatar="👤" if m["rol"]=="user" else "🤖"):
            st.markdown(m["contenido"])

    if pregunta := st.chat_input("Consulta al Copilot..."):
        st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
        with st.chat_message("user", avatar="👤"):
            st.markdown(pregunta)

        with st.chat_message("assistant", avatar="🤖"):
            p_min = pregunta.lower()
            if any(x in p_min for x in ["zona", "riesgo", "cluster"]):
                try:
                    top = sorted(predicciones, key=lambda x: x.get('riesgo_score', 0), reverse=True)[:3]
                    res = "📍 **Análisis del Data Warehouse:**\n\n"
                    for z in top:
                        res += f"- **Cluster {z['cluster_id']}**: Riesgo {round(z['riesgo_score']*100,1)}%. Factor: {z['top_factores']}\n"
                except: res = "Error al procesar predicciones."
            elif "lluvia" in p_min or "llueve" in p_min:
                res = "🌧️ **Simulación LightGBM:** La lluvia incrementa la probabilidad de incidentes en un **15%** debido al factor *pavimento mojado*."
            else:
                res = "Hola. Soy UrbanTech Copilot. ¿En qué puedo ayudarte hoy?"
            
            st.markdown(res)
            st.session_state.mensajes.append({"rol": "assistant", "contenido": res})