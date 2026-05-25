# -*- coding: utf-8 -*-
import streamlit as st
import json
import joblib
import pandas as pd
import numpy as np
from groq import Groq
from datetime import datetime
import streamlit.components.v1 as components
import base64

# ── configuracion ──────────────────────────────────────
st.set_page_config(
    page_title="UrbanTech Copilot",
    page_icon="🏙️",
    layout="wide"
)

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

# ── imagenes ───────────────────────────────────────────
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

# ── css ────────────────────────────────────────────────
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
        background-color: rgba(6, 15, 25, 0.95) !important;
        border-radius: 15px !important;
        padding: 25px !important;
        backdrop-filter: blur(12px);
    }}

    [data-testid="stColumn"]:nth-child(1) > div {{
        border: 1px solid rgba(0, 255, 255, 0.4) !important;
        box-shadow: 0 4px 15px rgba(0,255,255,0.1);
    }}
    [data-testid="stColumn"]:nth-child(2) > div {{
        border: 1px solid rgba(230, 126, 34, 0.4) !important;
        box-shadow: 0 4px 15px rgba(230,126,34,0.1);
    }}

    .context-box {{
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        font-size: 0.9em;
        line-height: 1.4;
    }}
    .bi-context {{
        background: rgba(0, 255, 255, 0.1);
        border-left: 4px solid #00ffff;
        color: #e0f7fa;
    }}
    .ai-context {{
        background: rgba(230, 126, 34, 0.1);
        border-left: 4px solid #e67e22;
        color: #fff3e0;
    }}

    .stChatInputContainer {{
        border: 1px solid #e67e22 !important;
        background: #0a1929 !important;
    }}

    [data-testid="stChatMessage"] {{
        background-color: transparent !important;
        border: 2px solid #e67e22 !important;
        border-radius: 10px;
        padding: 15px !important;
        margin-bottom: 15px;
    }}

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] ul,
    [data-testid="stChatMessage"] ol {{
        color: #ffffff !important;
        font-weight: 400 !important;
        font-size: 1.05em !important;
    }}

    [data-testid="stChatMessage"] strong {{
        color: #e67e22 !important;
        font-weight: 700 !important;
    }}
</style>
""", unsafe_allow_html=True)

# ── header ─────────────────────────────────────────────
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

# ── cargar modelo y datos ──────────────────────────────
@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("modelo_lightgbm_v20260508_2005.joblib")
    with open("metadata.json", "r") as f:
        meta = json.load(f)
    return modelo, meta

@st.cache_data
def cargar_predicciones():
    with open("predicciones_2024_con_clusters.json", "r", encoding="utf-8") as f:
        datos = json.load(f)
    datos_limpios = []
    for d in datos:
        datos_limpios.append({
            "celda_id": d.get("celda_id", ""),
            "alcaldia": d.get("contexto_espacial", {}).get("alcaldia", "sin dato"),
            "riesgo_score": round(d.get("riesgo_score", 0), 4),
            "clase_predicha": d.get("clase_predicha", 0),
            "nodos_riesgo": d.get("contexto_espacial", {}).get("nodos_riesgo", 0),
            "distancia_metro_m": round(d.get("contexto_espacial", {}).get("distancia_metro_m", 0), 0),
            "estacion_metro": d.get("contexto_espacial", {}).get("estacion_metro_cercana", "sin dato"),
            "n_bares": d.get("contexto_espacial", {}).get("n_bares", 0),
            "n_escuelas": d.get("contexto_espacial", {}).get("n_escuelas", 0),
            "top_factores": d.get("top_factores", []),
            "cluster_id": d.get("cluster_id", -1),
            "incidente_mas_comun": d.get("incidente_mas_comun", "sin historial"),
            "hora_pico": d.get("hora_pico", 0),
            "dia_pico": d.get("dia_pico", "sin historial"),
            "total_incidentes": d.get("total_incidentes", 0),
            "clima_dominante": d.get("clima_dominante", "sin historial")
        })
    return datos_limpios

modelo, meta = cargar_modelo()
datos = cargar_predicciones()

# ── prediccion con joblib ──────────────────────────────
ALCALDIA_MAP = {
    "cuauhtemoc": 3, "iztapalapa": 7, "tlalpan": 14,
    "iztacalco": 6, "benito juarez": 2, "alvaro obregon": 0,
    "venustiano carranza": 15, "miguel hidalgo": 9,
    "xochimilco": 16, "coyoacan": 4, "azcapotzalco": 1,
    "gustavo a madero": 5, "milpa alta": 10, "tlahuac": 13,
    "magdalena contreras": 8
}

def predecir_zona(alcaldia, hora, fecha, precipitacion=0,
                  temperatura=18, es_quincena=0, es_festivo=0):
    try:
        fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
        dia_semana = fecha_dt.weekday()
        dia_rad = 2 * np.pi * dia_semana / 7
        hora_rad = 2 * np.pi * hora / 24

        if hora < 6: turno = 0
        elif hora < 12: turno = 1
        elif hora < 18: turno = 2
        else: turno = 3

        hora_pico_flag = 1 if hora in range(7,10) or hora in range(16,20) else 0
        fin_de_semana_flag = 1 if dia_semana >= 5 else 0

        zona_ref = next((d for d in datos if d["alcaldia"] == alcaldia.lower()), None)
        nodos_riesgo = zona_ref["nodos_riesgo"] if zona_ref else 5
        distancia_metro = zona_ref["distancia_metro_m"] if zona_ref else 1000
        n_bares = zona_ref["n_bares"] if zona_ref else 0
        n_escuelas = zona_ref["n_escuelas"] if zona_ref else 0
        afluencia = 0.8
        cat_afluencia = 1
        lluvia_bin = 1 if precipitacion > 2 else 0

        fila = {
            "hora_creacion_int": hora,
            "hora_creacion_sin": np.sin(hora_rad),
            "hora_creacion_cos": np.cos(hora_rad),
            "dia_semana_num": dia_semana,
            "dia_semana_sin": np.sin(dia_rad),
            "dia_semana_cos": np.cos(dia_rad),
            "es_festivo": es_festivo,
            "es_quincena": es_quincena,
            "es_vacaciones_escolares": 0,
            "hora_pico_flag": hora_pico_flag,
            "fin_de_semana_flag": fin_de_semana_flag,
            "eventos_masivos_flag": 0,
            "turno": turno,
            "n_bares": n_bares,
            "n_escuelas": n_escuelas,
            "n_teatros": 0,
            "n_terminales": 0,
            "nodos_riesgo": nodos_riesgo,
            "distancia_metro_m": distancia_metro,
            "alcaldia_inicio": ALCALDIA_MAP.get(alcaldia.lower(), 3),
            "precipitacion": precipitacion,
            "temperatura": temperatura,
            "cobertura_nubes": 80 if precipitacion > 2 else 20,
            "indice_afluencia": afluencia,
            "categoria_afluencia": cat_afluencia,
            "int_lluvia_x_afluencia": lluvia_bin * afluencia,
            "int_horapico_x_lluvia": hora_pico_flag * lluvia_bin,
            "int_horapico_x_afluencia": hora_pico_flag * afluencia,
            "int_finde_x_eventos": fin_de_semana_flag * 0,
            "int_quincena_x_finde": es_quincena * fin_de_semana_flag
        }

        df_pred = pd.DataFrame([fila])
        prob = modelo.predict_proba(df_pred)[0][1]
        return {
            "alcaldia": alcaldia,
            "fecha": fecha,
            "hora": hora,
            "riesgo_score": round(prob, 4),
            "clase_predicha": 1 if prob > 0.5 else 0,
            "es_quincena": es_quincena,
            "precipitacion": precipitacion
        }
    except Exception as e:
        return {"error": str(e)}

def predecir_todas_alcaldias(fecha, hora, precipitacion=0, es_quincena=0):
    resultados = []
    for alcaldia in ALCALDIA_MAP.keys():
        res = predecir_zona(alcaldia, hora, fecha, precipitacion, 18, es_quincena)
        if "error" not in res:
            resultados.append(res)
    return sorted(resultados, key=lambda x: x["riesgo_score"], reverse=True)

# ── filtro dinamico ────────────────────────────────────
def filtrar_predicciones(datos, pregunta):
    pregunta_lower = pregunta.lower()
    for alcaldia in ALCALDIA_MAP.keys():
        if alcaldia in pregunta_lower:
            filtrados = [d for d in datos if d["alcaldia"] == alcaldia]
            if filtrados:
                return sorted(filtrados, key=lambda x: x["riesgo_score"], reverse=True)[:20]
    if any(p in pregunta_lower for p in ["peligrosas", "riesgo", "criticas", "zonas"]):
        return sorted([d for d in datos if d["clase_predicha"] == 1],
                     key=lambda x: x["riesgo_score"], reverse=True)[:20]
    return sorted(datos, key=lambda x: x["riesgo_score"], reverse=True)[:20]

def detectar_modo(pregunta):
    palabras_futuro = [
        "pasaria", "si lloviera", "que pasaria", "imagina",
        "simula", "pronostico", "2025", "2026", "2027",
        "mañana", "este viernes", "este lunes", "proxima semana"
    ]
    for palabra in palabras_futuro:
        if palabra in pregunta.lower():
            return "joblib"
    return "json"

# ── system prompt ──────────────────────────────────────
SYSTEM_PROMPT_BASE = """
Eres UrbanTech Copilot, agente experto en análisis de 
siniestralidad vial y planeación urbana para la CDMX.

Tu modelo LightGBM predice riesgo de incidentes en celdas 
de 1km² x 4 horas, entrenado con datos 2019-2023 y 
validado contra oct 2023 - feb 2024 con 78% de recall.

{CONTEXTO}

REGLAS:
1. CONSULTAS HISTÓRICAS: Ordena por riesgo_score, explica
   con SHAP values. SOLO zonas con celda_id real.
2. CONSULTAS FUTURAS: Presenta top 3 zonas con riesgo_score
   real del modelo. Explica condiciones.
3. EXPLICABILIDAD: SHAP positivo = aumenta riesgo,
   negativo = reduce. Conciso y accionable.
4. NUNCA inventes zonas. Máximo 300 palabras.

Responde en español, profesional y directo.
"""

# ── columnas ───────────────────────────────────────────
col_dash, col_chat = st.columns([1.6, 1], gap="large")

with col_dash:
    st.markdown("<h2 style='color:#00ffff !important;'>📊 Inteligencia de Negocios (BI)</h2>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class="context-box bi-context">
        <b style="color:#00ffff;">El Pulso del Caos:</b> Radiografía de incidentes en CDMX.<br>
        Explora el <i>Data Warehouse</i> histórico interactivo directamente en el tablero integrado.
    </div>
    """, unsafe_allow_html=True)

    looker_url = "https://lookerstudio.google.com/embed/reporting/6170d005-664f-4ba1-a21f-a3127ab0e52d/page/gWYvF"
    st.link_button("↗️ ABRIR DASHBOARD COMPLETO",
                  looker_url, type="primary", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)
    components.iframe(looker_url, height=700, scrolling=True)

with col_chat:
    c_titulo, c_boton = st.columns([0.7, 0.3])
    with c_titulo:
        st.markdown("<h2 style='color:#e67e22 !important;'>🤖 Agente XAI</h2>",
                   unsafe_allow_html=True)
    with c_boton:
        if st.button("🧹 Limpiar", use_container_width=True):
            st.session_state.mensajes = []
            st.rerun()

    st.markdown("""
    <div class="context-box ai-context">
        <b style="color:#e67e22;">IA al Servicio del Analista:</b><br>
        Consulta predicciones espaciales, factores de riesgo
        y explicaciones basadas en valores SHAP.
    </div>
    """, unsafe_allow_html=True)

    # preguntas sugeridas
    q1, q2 = st.columns(2)
    with q1:
        if st.button("🔴 3 zonas más peligrosas", use_container_width=True):
            st.session_state.pregunta_rapida = "¿Cuáles son las 3 zonas con mayor probabilidad de incidentes?"
        if st.button("🔍 Variables en Iztacalco", use_container_width=True):
            st.session_state.pregunta_rapida = "Explícame qué variables exógenas están detonando el riesgo en Iztacalco"
    with q2:
        if st.button("📅 Viernes de quincena", use_container_width=True):
            st.session_state.pregunta_rapida = "¿Cuáles son las 3 zonas más peligrosas este viernes siendo quincena?"
        if st.button("🌧️ Impacto de lluvia", use_container_width=True):
            st.session_state.pregunta_rapida = "¿Cómo afecta la lluvia al riesgo en las zonas críticas?"

    st.divider()

    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    for msg in st.session_state.mensajes:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    pregunta = st.chat_input("Consulta al Copilot...")

    if "pregunta_rapida" in st.session_state:
        pregunta = st.session_state.pregunta_rapida
        del st.session_state.pregunta_rapida

    if pregunta:
        with st.chat_message("user", avatar="👤"):
            st.markdown(pregunta)
        st.session_state.mensajes.append({"role": "user", "content": pregunta})

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analizando..."):
                modo = detectar_modo(pregunta)

                if modo == "joblib":
                    hoy = datetime.now()
                    fecha = hoy.strftime("%Y-%m-%d")
                    es_quincena = 1 if hoy.day in [1, 15] else 0
                    precipitacion = 8.0 if any(
                        p in pregunta.lower() for p in
                        ["lluvia", "llueve", "lloviera", "tormenta"]
                    ) else 0.0

                    predicciones_nuevas = predecir_todas_alcaldias(
                        fecha=fecha, hora=17,
                        precipitacion=precipitacion,
                        es_quincena=es_quincena
                    )
                    contexto = f"""
MODO: Predicción en tiempo real con modelo LightGBM
Fecha: {fecha} | Quincena: {"Sí" if es_quincena else "No"} | Lluvia: {precipitacion}mm | Hora: 17:00

Predicciones del modelo:
{json.dumps(predicciones_nuevas[:10], ensure_ascii=False)}
"""
                else:
                    predicciones_filtradas = filtrar_predicciones(datos, pregunta)
                    contexto = f"""
MODO: Consulta histórica del Data Warehouse (oct 2023 - feb 2024)
{json.dumps(predicciones_filtradas, ensure_ascii=False)}
"""

                system_prompt = SYSTEM_PROMPT_BASE.replace("{CONTEXTO}", contexto)

                historial_texto = ""
                for msg in st.session_state.mensajes[:-1]:
                    rol = "Usuario" if msg["role"] == "user" else "Copilot"
                    historial_texto += f"{rol}: {msg['content']}\n"

                respuesta = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content":
                         f"Historial:\n{historial_texto}\n\nPregunta: {pregunta}"}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                texto = respuesta.choices[0].message.content
                st.markdown(texto)

        st.session_state.mensajes.append({"role": "assistant", "content": texto})