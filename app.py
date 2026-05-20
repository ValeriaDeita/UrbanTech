# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import json
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

# --- 3. CSS MAESTRO ---
st.markdown(f"""
{fondo_style}
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Orbitron:wght@500;900&display=swap');

    body, .stApp {{ font-family: 'Inter', sans-serif; color: white; }}
    [data-testid="stHeader"], [data-testid="stToolbar"] {{ display: none !important; }}

    .hero-header {{
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(4, 15, 26, 0.95);
        padding: 40px;
        border-radius: 20px;
        border-bottom: 3px solid #e67e22;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.8);
        border: 1px solid rgba(230, 126, 34, 0.3);
    }}
    .logo-container img {{ height: 150px; filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.5)); }}
    .header-text h1 {{ font-family: 'Orbitron', sans-serif; font-size: 3.5em !important; color: #ffffff !important; margin: 0; letter-spacing: 2px; }}

    [data-testid="stColumn"] > div {{ background-color: rgba(6, 15, 25, 0.95) !important; border-radius: 15px !important; padding: 25px !important; backdrop-filter: blur(12px); }}
    [data-testid="stColumn"]:nth-child(1) > div {{ border: 1px solid rgba(0, 255, 255, 0.4) !important; }}
    [data-testid="stColumn"]:nth-child(2) > div {{ border: 1px solid rgba(230, 126, 34, 0.4) !important; }}

    .context-box {{ padding: 15px; border-radius: 10px; margin-bottom: 20px; font-size: 0.9em; }}
    .bi-context {{ background: rgba(0, 255, 255, 0.1); border-left: 4px solid #00ffff; color: #e0f7fa; }}
    .ai-context {{ background: rgba(230, 126, 34, 0.1); border-left: 4px solid #e67e22; color: #fff3e0; }}

    /* ESTILO DE CAJAS DE CHAT */
    [data-testid="stChatMessage"] {{
        background-color: rgba(230, 126, 34, 0.15) !important;
        border: 2px solid #e67e22 !important;
        border-radius: 10px;
        padding: 15px !important;
        margin-bottom: 15px;
    }}
    [data-testid="stChatMessage"], [data-testid="stChatMessage"] * {{
        color: #ffffff !important;
        font-size: 1.05em !important;
    }}
    [data-testid="stChatMessage"] strong, [data-testid="stChatMessage"] b {{
        color: #e67e22 !important;
        font-weight: 900 !important;
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
        <p style='color: #00ffff; font-weight: bold;'>MINERÍA DE DATOS | FES ACATLÁN</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. LÓGICA DEL AGENTE ---
@st.cache_resource
def inicializar_ia():
    with open('predicciones_2024_con_clusters.json', 'r', encoding='utf-8') as f:
        predicciones = json.load(f)
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    contexto_datos = json.dumps(predicciones[:5])
    instruccion = f"""
    Eres UrbanTech Copilot, agente de IA Explicable.
    Funciones:
    1. **Predicción de Riesgo**: Calculas **riesgo_score** y **clase_predicha**.
    2. **Hotspots**: Identificas zonas de alta siniestralidad usando **valores SHAP**.
    Reglas: Responde siempre resaltando términos clave como **riesgo_score**, **clase_predicha** o **nodos de riesgo** en negritas. Datos: {contexto_datos}.
    """
    return genai.GenerativeModel('gemini-1.5-flash', system_instruction=instruccion)

modelo_gemini = inicializar_ia()

# --- 6. INTERFAZ ---
col_dash, col_chat = st.columns([1.6, 1], gap="large")
with col_dash:
    st.markdown("<h2 style='color:#00ffff;'>📊 Inteligencia de Negocios (BI)</h2>", unsafe_allow_html=True)
    looker_url = "https://lookerstudio.google.com/embed/reporting/6170d005-664f-4ba1-a21f-a3127ab0e52d/page/gWYvF"
    st.link_button("↗️ ABRIR DASHBOARD COMPLETO", looker_url, type="primary", use_container_width=True)
    components.iframe(looker_url, height=700, scrolling=True)

with col_chat:
    st.markdown("<h2 style='color:#e67e22;'>🤖 Agente XAI</h2>", unsafe_allow_html=True)
    if "mensajes_ui" not in st.session_state: st.session_state.mensajes_ui = []
    if "chat_session" not in st.session_state: st.session_state.chat_session = modelo_gemini.start_chat(history=[])

    for m in st.session_state.mensajes_ui:
        with st.chat_message(m["rol"], avatar="👤" if m["rol"]=="user" else "🤖"):
            st.markdown(m["contenido"])

    if pregunta := st.chat_input("Consulta al Copilot..."):
        st.session_state.mensajes_ui.append({"rol": "user", "contenido": pregunta})
        with st.chat_message("user", avatar="👤"): st.markdown(pregunta)
        with st.chat_message("assistant", avatar="🤖"):
            respuesta = st.session_state.chat_session.send_message(pregunta).text
            st.markdown(respuesta)
            st.session_state.mensajes_ui.append({"rol": "assistant", "contenido": respuesta})