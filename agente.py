

import streamlit as st
import json
import joblib
import pandas as pd
import numpy as np
from groq import Groq
from datetime import datetime

# ── configuracion ──────────────────────────────────────

GROQ_API_KEY = st.secrets["GROQ_API_KEY"] 
client = Groq(api_key=GROQ_API_KEY)

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

CATEGORIA_AFLUENCIA_MAP = {"baja": 0, "normal": 1, "alta": 2, "critica": 3}
TURNO_MAP = {"madrugada": 0, "manana": 1, "tarde": 2, "noche": 3}

def predecir_zona(alcaldia, hora, fecha, precipitacion=0,
                  temperatura=18, es_quincena=0, es_festivo=0):
    try:
        fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
        dia_semana = fecha_dt.weekday()
        dia_rad = 2 * np.pi * dia_semana / 7
        hora_rad = 2 * np.pi * hora / 24

        if hora < 6:
            turno = 0
        elif hora < 12:
            turno = 1
        elif hora < 18:
            turno = 2
        else:
            turno = 3

        hora_pico_flag = 1 if hora in range(7, 10) or hora in range(16, 20) else 0
        fin_de_semana_flag = 1 if dia_semana >= 5 else 0

        # buscar contexto espacial de la alcaldia en el JSON
        zona_ref = next(
            (d for d in datos if d["alcaldia"] == alcaldia.lower()),
            None
        )
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
        clase = 1 if prob > 0.5 else 0

        return {
            "alcaldia": alcaldia,
            "fecha": fecha,
            "hora": hora,
            "riesgo_score": round(prob, 4),
            "clase_predicha": clase,
            "es_quincena": es_quincena,
            "precipitacion": precipitacion
        }
    except Exception as e:
        return {"error": str(e)}

def predecir_todas_alcaldias(fecha, hora, precipitacion=0,
                              es_quincena=0, es_festivo=0):
    resultados = []
    for alcaldia in ALCALDIA_MAP.keys():
        res = predecir_zona(
            alcaldia, hora, fecha,
            precipitacion, 18, es_quincena, es_festivo
        )
        if "error" not in res:
            resultados.append(res)
    return sorted(resultados,
                 key=lambda x: x["riesgo_score"],
                 reverse=True)

# ── filtro dinamico ────────────────────────────────────
ALCALDIAS = list(ALCALDIA_MAP.keys())

def filtrar_predicciones(datos, pregunta):
    pregunta_lower = pregunta.lower()
    for alcaldia in ALCALDIAS:
        if alcaldia in pregunta_lower:
            filtrados = [d for d in datos
                        if d["alcaldia"] == alcaldia]
            if filtrados:
                return sorted(filtrados,
                             key=lambda x: x["riesgo_score"],
                             reverse=True)[:20]
    if any(p in pregunta_lower for p in
           ["peligrosas", "riesgo", "criticas", "zonas"]):
        return sorted(
            [d for d in datos if d["clase_predicha"] == 1],
            key=lambda x: x["riesgo_score"],
            reverse=True
        )[:20]
    return sorted(datos,
                 key=lambda x: x["riesgo_score"],
                 reverse=True)[:20]

def detectar_modo(pregunta):
    palabras_futuro = [
        "pasaria", "si lloviera", "que pasaria",
        "imagina", "simula", "pronostico",
        "2025", "2026", "2027", "mañana",
        "este viernes", "este lunes", "proxima semana"
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

1. CONSULTAS HISTÓRICAS (datos del JSON):
   - Ordena por riesgo_score descendente
   - Explica usando top_factores y SHAP values
   - SHAP positivo = factor que AUMENTA el riesgo
   - SHAP negativo = factor que REDUCE el riesgo
   - SOLO menciona zonas con celda_id real de los datos

2. CONSULTAS FUTURAS (predicciones nuevas del modelo):
   - Se te proporcionan predicciones reales del modelo
   - Presenta las top 3 zonas con mayor riesgo_score
   - Explica qué condiciones generan ese riesgo
   - Menciona si es quincena, lluvia o hora pico

3. EXPLICABILIDAD:
   - Traduce valores a lenguaje natural
   - Se conciso y accionable
   - Menciona alcaldia y factores clave

4. LIMITACIONES:
   - NUNCA inventes zonas que no estén en los datos
   - Máximo 300 palabras por respuesta

Responde en español de forma profesional y directa.
"""

# ── interfaz ───────────────────────────────────────────
st.set_page_config(
    page_title="UrbanTech Copilot",
    page_icon="🏙️",
    layout="centered"
)

st.title("🏙️ UrbanTech Copilot")
st.caption("Agente de inteligencia urbana para la CDMX")

col1, col2 = st.columns(2)
with col1:
    if st.button("¿Cuáles son las 3 zonas más peligrosas?"):
        st.session_state.pregunta_rapida = "¿Cuáles son las 3 zonas con mayor probabilidad de incidentes?"
    if st.button("¿Qué variables detonan el riesgo en Iztacalco?"):
        st.session_state.pregunta_rapida = "Explícame qué variables exógenas están detonando el riesgo en Iztacalco"
with col2:
    if st.button("¿Zonas de riesgo este viernes de quincena?"):
        st.session_state.pregunta_rapida = "¿Cuáles son las 3 zonas más peligrosas este viernes siendo quincena?"
    if st.button("¿Afecta la lluvia al riesgo?"):
        st.session_state.pregunta_rapida = "¿Cómo afecta la lluvia al riesgo en las zonas críticas?"

st.divider()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for msg in st.session_state.mensajes:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

pregunta = st.chat_input("Consulta al Copilot...")

if "pregunta_rapida" in st.session_state:
    pregunta = st.session_state.pregunta_rapida
    del st.session_state.pregunta_rapida

if pregunta:
    with st.chat_message("user"):
        st.write(pregunta)
    st.session_state.mensajes.append(
        {"role": "user", "content": pregunta}
    )

    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            modo = detectar_modo(pregunta)

            if modo == "joblib":
                # detectar fecha y condiciones de la pregunta
                hoy = datetime.now()
                fecha = hoy.strftime("%Y-%m-%d")

                # detectar si es quincena
                es_quincena = 1 if hoy.day in [1, 15] else 0

                # detectar lluvia
                precipitacion = 8.0 if any(
                    p in pregunta.lower() for p in
                    ["lluvia", "llueve", "lloviera", "tormenta"]
                ) else 0.0

                # generar predicciones reales con el modelo
                predicciones_nuevas = predecir_todas_alcaldias(
                    fecha=fecha,
                    hora=17,
                    precipitacion=precipitacion,
                    es_quincena=es_quincena
                )

                contexto = f"""
MODO: Predicción en tiempo real con modelo LightGBM

Fecha consultada: {fecha}
Es quincena: {"Sí" if es_quincena else "No"}
Precipitación: {precipitacion}mm
Hora analizada: 17:00 hrs

Predicciones generadas por el modelo para TODAS las alcaldías:
{json.dumps(predicciones_nuevas[:10], ensure_ascii=False)}
"""
            else:
                predicciones_filtradas = filtrar_predicciones(
                    datos, pregunta
                )
                contexto = f"""
MODO: Consulta histórica del Data Warehouse

Predicciones relevantes del período oct 2023 - feb 2024:
{json.dumps(predicciones_filtradas, ensure_ascii=False)}
"""

            system_prompt = SYSTEM_PROMPT_BASE.replace(
                "{CONTEXTO}", contexto
            )

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
            st.write(texto)

    st.session_state.mensajes.append(
        {"role": "assistant", "content": texto}
    )