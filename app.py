# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json
import joblib
import streamlit.components.v1 as components
import base64
import google.generativeai as genai

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

    body, .stApp {{ font-family: 'Inter', sans-serif; }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{ display: none !important; }}

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
        height: 150px;
        filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.5));
    }}
    .header-text {{ margin-left: 30px; text-align: left; }}
    .header-text h1 {{
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5em !important;
        color: #ffffff !important;
        margin: 0;
        letter-spacing: 2px;
    }}

    [data-testid="stColumn"] > div {{
        background-color: rgba(6, 15, 25, 0.92) !important;
        border-radius: 15px !important;
        padding: 25px !important;
        backdrop-filter: blur(10px);
    }}
    
    [data-testid="stColumn"]:nth-child(1) > div {{ border: 1px solid #00ffff !important; }}
    [data-testid="stColumn"]:nth-child(2) > div {{ border: 1px solid #e67e22 !important; }}

    .context-box {{
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 0.9em;
        line-height: 1.4;
    }}
    .bi-context {{ background: rgba(0, 255, 255, 0.05); border-left: 4px solid #00ffff; }}
    .ai-context {{ background: rgba(230, 126, 34, 0.05); border-left: 4px solid #e67e22; }}

    .stChatInputContainer {{ border: 1px solid #e67e22 !important; background: #0a1929 !important; }}
</style>
""", unsafe_allow_html=True)

# --- 4. RENDERIZADO DEL ENCABEZADO ---
logo_img_html = f'<img src="data:image/png;base64,{logo_b64}">' if logo_b64 else "🏙️"
st.markdown(f"""
<div class="hero-header">
    <div class="logo-container">{logo_img_html}</div>
    <div class="header-text">
        <h1>UrbanTech Copilot</h1>
        <p style='color: #00ffff; font-weight: bold; margin:0; font-size: 1.1em; letter-spacing: 1px;'>MINERÍA DE DATOS | FES ACATLÁN</p>
        <p style='color: #888; margin:0; font-size: 0.9em; margin-top: 5px;'>Equipo: Denisse, Alan, Josua, Samuel, Deita, Jesus, Frida</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. CARGA DE DATOS Y BÚSQUEDA AUTOMÁTICA DE MODELO ---
@st.cache_resource
def inicializar_ia():
    with open('predicciones_2024_con_clusters.json', 'r', encoding='utf-8') as f:
        predicciones = json.load(f)
    
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    
    # Buscador dinámico: Encuentra el primer modelo válido en tu cuenta
    modelo_valido = 'models/gemini-1.5-flash' # Fallback por defecto
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods and 'flash' in m.name:
            modelo_valido = m.name
            break

    contexto_datos = json.dumps(predicciones[:5])
    instruccion = f"""
    Eres UrbanTech Copilot, un agente de Inteligencia Artificial para análisis de siniestralidad vial y planeación urbana.
    Tu modelo se considera una 'Distribution of Flow'.
    Datos actuales del Data Warehouse (clusters de riesgo): {contexto_datos}.
    Responde preguntas sobre hotspots y simula impactos climáticos de forma analítica, profesional y concisa.
    """
    
    modelo = genai.GenerativeModel(modelo_valido, system_instruction=instruccion)
    return modelo

try:
    modelo_gemini = inicializar_ia()
    status_ready = True
except Exception as e:
    status_ready = False
    error_msg = e

# --- 6. COLUMNAS ---
col_dash, col_chat = st.columns([1.6, 1], gap="large")

with col_dash:
    st.markdown("<h2 style='color:#00ffff !important;'>📊 Inteligencia de Negocios (BI)</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="context-box bi-context">
        <b style="color:#00ffff;">El Pulso del Caos:</b> Radiografía de incidentes en CDMX.<br>
        Este tablero procesa el <i>Data Warehouse</i> histórico para identificar patrones espaciales. 
    </div>
    """, unsafe_allow_html=True)
    
    looker_url = "https://lookerstudio.google.com/embed/reporting/6170d005-664f-4ba1-a21f-a3127ab0e52d/page/gWYvF"
    components.iframe(looker_url, height=750, scrolling=True)

with col_chat:
    st.markdown("<h2 style='color:#e67e22 !important;'>🤖 Agente Predictivo (IA)</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="context-box ai-context">
        <b style="color:#e67e22;">IA al Servicio del Analista:</b><br>
        Consulta datos espaciales o simula escenarios climáticos.
    </div>
    """, unsafe_allow_html=True)

    if status_ready:
        st.success("✅ Motor de Inteligencia Artificial activo.")
    else:
        st.error(f"⚠️ Error: {error_msg}")

    if "chat_session" not in st.session_state and status_ready:
        st.session_state.chat_session = modelo_gemini.start_chat(history=[])

    if "mensajes_ui" not in st.session_state:
        st.session_state.mensajes_ui = []

    for m in st.session_state.mensajes_ui:
        with st.chat_message(m["rol"], avatar="👤" if m["rol"]=="user" else "🤖"):
            st.markdown(m["contenido"])

    if pregunta := st.chat_input("Consulta al Copilot..."):
        st.session_state.mensajes_ui.append({"rol": "user", "contenido": pregunta})
        with st.chat_message("user", avatar="👤"):
            st.markdown(pregunta)

        with st.chat_message("assistant", avatar="🤖"):
            if status_ready:
                with st.spinner("Analizando..."):
                    try:
                        respuesta_ia = st.session_state.chat_session.send_message(pregunta)
                        texto_respuesta = respuesta_ia.text
                    except Exception as e:
                        texto_respuesta = f"Error al generar respuesta: {str(e)}"
                    
                    st.markdown(texto_respuesta)
                    st.session_state.mensajes_ui.append({"rol": "assistant", "contenido": texto_respuesta})