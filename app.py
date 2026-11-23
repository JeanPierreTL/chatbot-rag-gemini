"""
Chatbot RAG sobre documentos PDF
Gemini (embeddings + LLM) · MongoDB Atlas Vector Search · Streamlit Community Cloud

Curso: Tecnologías Emergentes – Semana 10

Las credenciales NO están en este archivo: se leen de st.secrets
(Streamlit Cloud → Settings → Secrets, o .streamlit/secrets.toml en local).

    [app]
    GOOGLE_API_KEY = "<su_api_key>"
    MONGODB_URI    = "mongodb+srv://<usuario>:<password>@<cluster>.mongodb.net/"
"""

import time

import pandas as pd
import pymongo
import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Configuración (para la TAREA EXTRA basta con cambiar este bloque)
# ---------------------------------------------------------------------------
TITULO_APP       = "Chatbot RAG · AWS Cloud Adoption Framework"
DESCRIPCION_APP  = "Pregunte sobre el documento *AWS Cloud Adoption Framework* (PDF de clase)."
BASE_DATOS       = "pdf_embeddings_db"
COLECCION        = "pdf_vectors"
INDICE           = "vector_index"
MODELO_EMBEDDING = "gemini-embedding-001"   # 3072 dimensiones
# gemini-2.5-flash ya no está disponible para cuentas nuevas: se prueban estos
# modelos en orden y se pasa al siguiente si uno está saturado (503) o no existe (404).
MODELOS_LLM      = ["gemini-3.1-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash"]
K_POR_DEFECTO    = 5                        # fragmentos recuperados (limit)
NUM_CANDIDATOS   = 100                      # candidatos evaluados por $vectorSearch
ROL_ASISTENTE    = "un asistente experto en adopción de la nube"

# ---------------------------------------------------------------------------
# Paso 8. Secretos
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI    = st.secrets["app"]["MONGODB_URI"]


# ---------------------------------------------------------------------------
# Paso 9. Clientes en caché: se crean una sola vez y se reutilizan entre
# las re-ejecuciones del script que Streamlit hace en cada interacción.
# ---------------------------------------------------------------------------
@st.cache_resource
def obtener_cliente_genai():
    return genai.Client(api_key=GOOGLE_API_KEY, http_options=types.HttpOptions(timeout=40000))  # 40 s máx. por llamada


@st.cache_resource
def obtener_coleccion():
    cliente = pymongo.MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
    return cliente[BASE_DATOS][COLECCION]


client_genai = obtener_cliente_genai()
collection   = obtener_coleccion()


# ---------------------------------------------------------------------------
# Paso 10. Recuperación semántica
# ---------------------------------------------------------------------------
def crear_embedding_consulta(texto: str) -> list[float]:
    """La pregunta se vectoriza con task_type=RETRIEVAL_QUERY (con reintentos si Gemini está saturado)."""
    for intento in range(4):
        try:
            resp = client_genai.models.embed_content(
                model=MODELO_EMBEDDING,
                contents=texto,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            return resp.embeddings[0].values
        except Exception:
            if intento == 3:
                raise
            time.sleep(2 * (intento + 1))


def buscar_similares(pregunta: str, k: int = K_POR_DEFECTO) -> list[dict]:
    """Devuelve los k fragmentos más cercanos con su score."""
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDICE,
                "path": "embedding",
                "queryVector": crear_embedding_consulta(pregunta),
                "numCandidates": max(NUM_CANDIDATOS, k * 10),
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "id": 1,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    return list(collection.aggregate(pipeline))


# ---------------------------------------------------------------------------
# Paso 11. Generación de la respuesta
# ---------------------------------------------------------------------------
def generar_respuesta(pregunta: str, fragmentos: list[dict]) -> str:
    if not fragmentos:
        return "No encontré información relevante en el documento para responder esa pregunta."

    contexto = "\n\n".join(f["texto"] for f in fragmentos)
    prompt = f"""Eres {ROL_ASISTENTE}. Usa EXCLUSIVAMENTE el siguiente contexto para responder la pregunta del usuario. Si la respuesta no está en el contexto, indícalo claramente.

Contexto:
{contexto}

Pregunta: {pregunta}

Responde de forma concisa y clara en español."""

    ultimo_error = None
    for ronda in range(3):                      # hasta 3 rondas sobre la lista de modelos
        for modelo in MODELOS_LLM:
            try:
                resp = client_genai.models.generate_content(model=modelo, contents=prompt)
                return resp.text
            except Exception as e:  # 503 saturado, 404 no disponible, 429 cuota
                ultimo_error = e
        time.sleep(3 * (ronda + 1))
    raise ultimo_error


# ---------------------------------------------------------------------------
# Paso 12. Interfaz de chat
# ---------------------------------------------------------------------------
st.set_page_config(page_title=TITULO_APP, page_icon="📄", layout="centered")
st.title(TITULO_APP)
st.caption(DESCRIPCION_APP)

with st.sidebar:
    st.header("Parámetros")
    k = st.select_slider(
        "Fragmentos recuperados (k)",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        value=K_POR_DEFECTO,
        help="Pruebe k = 3, 5 y 10 para el análisis de la actividad.",
    )
    st.write(f"Colección: `{BASE_DATOS}.{COLECCION}`")
    st.write(f"Índice: `{INDICE}`")
    if st.button("Borrar conversación"):
        st.session_state.historial = []
        st.rerun()

if "historial" not in st.session_state:
    st.session_state.historial = []


def mostrar_fragmentos(fragmentos: list[dict]):
    with st.expander(f"Fragmentos recuperados ({len(fragmentos)})"):
        if not fragmentos:
            st.write("Sin resultados.")
            return
        tabla = pd.DataFrame(
            [{"pos.": i + 1, "id": f.get("id"), "score": round(f["score"], 4),
              "inicio del texto": f["texto"][:120].replace("\n", " ") + "…"}
             for i, f in enumerate(fragmentos)]
        )
        st.dataframe(tabla, hide_index=True, use_container_width=True)
        for i, f in enumerate(fragmentos, 1):
            st.markdown(f"**[{i}] id = {f.get('id')} · score = {f['score']:.4f}**")
            st.text(f["texto"])


# Repintar el historial (Streamlit re-ejecuta todo el script en cada interacción)
for msg in st.session_state.historial:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["contenido"])
        if msg.get("fragmentos") is not None:
            mostrar_fragmentos(msg["fragmentos"])

pregunta = st.chat_input("Escriba su pregunta sobre el documento…")
if pregunta:
    st.session_state.historial.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        fragmentos = []
        try:
            with st.spinner("Buscando en el documento y redactando la respuesta…"):
                fragmentos = buscar_similares(pregunta, k)
                respuesta = generar_respuesta(pregunta, fragmentos)
        except Exception as e:  # muestra el error sin exponer credenciales
            respuesta = (f"⚠️ Gemini no respondió ({type(e).__name__}). "
                         "Suele ser saturación temporal del servicio: intente de nuevo en unos segundos.")
        st.markdown(respuesta)
        mostrar_fragmentos(fragmentos)

    st.session_state.historial.append(
        {"rol": "assistant", "contenido": respuesta, "fragmentos": fragmentos}
    )
