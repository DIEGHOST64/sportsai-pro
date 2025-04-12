import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
import torch
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from speech_recognition import Recognizer, AudioFile
import plotly.express as px
import speech_recognition as sr  # Se importa para usar en la transcripción

# --------------------------
# 1. Configuración Principal
# --------------------------
st.set_page_config(
    page_title="🏆 SportsAI Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------
# 2. Configuración API y Secrets
# --------------------------
if "OPENROUTER_API_KEY" not in st.secrets:
    st.error("❌ API Key no encontrada. Configúrala en Secrets (⚡ icono)")
    st.stop()

client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
)

# --------------------------
# 3. Modelos y Embeddings (Cacheados)
# --------------------------
@st.cache_resource
def cargar_recursos():
    embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    # Frases para robustecer la detección en cada deporte
    frases_deportes = {
        "fútbol": [
            "futbol", "fútbol", "soccer", "balón pie", 
            "historia del fútbol", "cómo se juega fútbol",
            "reglas del fútbol", "normas del fútbol", "jugadores en fútbol"
        ],
        "baloncesto": [
            "baloncesto", "basketball", "básquet", 
            "qué es baloncesto", "cómo se juega basket"
        ],
        "voleibol": [
            "voleibol", "volleyball", "bola alta", "reglas de voleibol"
        ],
        "rugby": [
            "rugby", "deporte de contacto", "pelota ovalada", "cómo se juega rugby"
        ],
        "mma": [
            "mma", "artes marciales mixtas", "ufc", "peleas profesionales", "lucha mma"
        ]
    }

    return {
        deporte: [embedder.encode(frase, convert_to_tensor=True) for frase in frases]
        for deporte, frases in frases_deportes.items()
    }, embedder

embeddings_referencia, embedder = cargar_recursos()

# Diccionario con palabras clave directas para cada deporte
PALABRAS_DIRECTAS = {
    "fútbol": ["futbol", "fútbol", "soccer"],
    "baloncesto": ["baloncesto", "basketball", "básquet"],
    "voleibol": ["voleibol", "volleyball"],
    "rugby": ["rugby"],
    "mma": ["mma", "artes marciales mixtas", "ufc"]
}

# --------------------------
# 4. Detección de Deporte
# --------------------------
def detectar_deporte(mensaje):
    if not mensaje.strip():
        return None

    lower_message = mensaje.lower()
    # Verificación directa para cada deporte
    for deporte, palabras in PALABRAS_DIRECTAS.items():
        for palabra in palabras:
            if palabra.lower() in lower_message:
                return deporte

    # Si no hay coincidencia directa, usar embeddings
    mensaje_embedding = embedder.encode(mensaje, convert_to_tensor=True)
    mejor_score, mejor_deporte = 0, None
    scores = {}  # Opcional: almacenar puntajes para debug

    for deporte, embeddings in embeddings_referencia.items():
        for emb in embeddings:
            similitud = util.cos_sim(mensaje_embedding, emb).item()
            if deporte not in scores or similitud > scores[deporte]:
                scores[deporte] = similitud
            if similitud > mejor_score:
                mejor_score, mejor_deporte = similitud, deporte

    # Opcional: st.write("Puntajes de similitud:", scores)
    # Umbral ajustado para determinar si la consulta es suficientemente "deportiva"
    return mejor_deporte if mejor_score >= 0.60 else None

# --------------------------
# 5. Generación de Respuesta
# --------------------------
PETICIONES_DETALLE = [
    "más detalles", "explica extenso", "a fondo", 
    "detalla más", "información completa", "quiero saber más"
]

def generar_respuesta_mejorada(prompt, contexto):
    try:
        if contexto['detalle']:
            # Respuesta extensa y detallada
            sistema = f"""Eres un experto en {contexto['deporte']} con 20 años de experiencia.
Nivel usuario: {contexto['nivel']}.
Proporciona una respuesta clara, detallada y extensa, incluyendo ejemplos, estadísticas y explicaciones completas cuando sea posible."""
            temperatura = 0.7
            max_tokens = 600
        else:
            # Respuesta breve y concisa
            sistema = f"""Eres un experto en {contexto['deporte']} con 20 años de experiencia.
Nivel usuario: {contexto['nivel']}.
Proporciona una respuesta concisa y práctica."""
            temperatura = 0.4
            max_tokens = 400

        response = client.chat.completions.create(
            model="openai/gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": sistema},
                {"role": "user", "content": prompt}
            ],
            temperature=temperatura,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error("❌ Hubo un problema al generar la respuesta. Intenta nuevamente más tarde.")
        st.exception(e)
        return "⚠️ Error inesperado al generar la respuesta."

# --------------------------
# 6. Interfaz Sidebar
# --------------------------
def mostrar_sidebar():
    with st.sidebar:
        st.header("⚙️ Preferencias")
        nivel = st.selectbox("Nivel de conocimiento", ["básico", "intermedio", "avanzado"])
        deportes = st.multiselect("Deportes favoritos", list(embeddings_referencia.keys()))

        # Detectar dispositivos de entrada
        dispositivos = sd.query_devices()
        dispositivos_entrada = [
            (i, d['name']) for i, d in enumerate(dispositivos) if d['max_input_channels'] > 0
        ]
        nombres_dispositivos = [f"{nombre} (id: {i})" for i, nombre in dispositivos_entrada]
        seleccionado = st.selectbox("🎙️ Elegir micrófono", nombres_dispositivos)

        # Guardar ID del micrófono seleccionado
        id_microfono = int(seleccionado.split("id: ")[-1].replace(")", ""))
        st.session_state["mic_device_id"] = id_microfono

        if st.button("💾 Guardar Preferencias"):
            st.session_state.user_prefs = {
                "nivel": nivel,
                "deportes_favoritos": deportes
            }

        st.divider()
        if st.button("🗑️ Reiniciar Chat"):
            st.session_state.clear()
            st.rerun()

# --------------------------
# 7. Multimedia por Deporte
# --------------------------
def mostrar_multimedia(deporte):
    urls_video = {
        "fútbol": "https://youtu.be/qknP-E-vPQ4",
        "baloncesto": "https://youtu.be/XbtmGKif7Ck",
        "voleibol": "https://youtu.be/gNfU7R3mN-0",
        "rugby": "https://youtu.be/GOxFzJ4vU2g",
        "mma": "https://youtu.be/n4WxJFr9HyQ"
    }

    st.subheader(f"📺 Multimedia sobre {deporte.capitalize()}")
    with st.expander("🎥 Video Explicativo"):
        st.video(urls_video.get(deporte, "https://youtu.be/dQw4w9WgXcQ"))

    with st.expander("📊 Estadísticas"):
        fig = px.bar(
            x=["Acción 1", "Acción 2", "Acción 3"], 
            y=[45, 32, 12], 
            labels={'x': 'Métrica', 'y': 'Total'},
            title=f"Estadísticas generales de {deporte}"
        )
        st.plotly_chart(fig)

# --------------------------
# 8. Grabación y Transcripción
# --------------------------
def grabar_audio():
    """Graba audio desde el micrófono seleccionado por el usuario"""
    FRECUENCIA = 44100
    DURACION = 5

    try:
        dispositivo_entrada = st.session_state.get("mic_device_id", sd.default.device[0])
        nombre_dispositivo = sd.query_devices(dispositivo_entrada)['name']

        with st.status(f"🎤 Grabando desde: {nombre_dispositivo}... ¡Habla ahora!", expanded=True) as status:
            grabacion = sd.rec(int(DURACION * FRECUENCIA),
                               samplerate=FRECUENCIA,
                               channels=1,
                               dtype='int16',
                               device=dispositivo_entrada,
                               blocking=True)
            sd.wait()
            wav.write("temp.wav", FRECUENCIA, grabacion)
            status.update(label="✅ Audio grabado", state="complete")

        return "temp.wav"

    except Exception as e:
        st.error(f"⚠️ Error al grabar audio: {str(e)}")
        return None

def transcribir_audio():
    """Convierte el audio grabado a texto usando speech_recognition"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile("temp.wav") as source:
            audio = recognizer.record(source)
            texto = recognizer.recognize_google(audio, language="es-CO")
            return texto
    except Exception as e:
        st.error(f"⚠️ Error al transcribir el audio: {str(e)}")
        return None

# --------------------------
# 9. Interfaz Principal
# --------------------------
def main():
    mostrar_sidebar()

    col1, col2 = st.columns([3, 1])

    with col1:
        st.title("🤖 Asistente Deportivo Inteligente")

        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.contexto = {"deporte": None, "detalle": False}
            st.info("💡 Ejemplos: '¿Cuántos jugadores tiene el voleibol?' o 'explica a fondo las reglas del rugby'")

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        col_entrada, col_voz = st.columns([4, 1])
        prompt = ""

        with col_entrada:
            prompt = st.chat_input("Escribe tu pregunta deportiva...")

        with col_voz:
            if st.button("🎤 Usar voz", use_container_width=True):
                archivo_audio = grabar_audio()
                # Solo se transcribe si se pudo grabar el audio
                if archivo_audio:
                    prompt = transcribir_audio()

        if prompt:
            # Detectar si la consulta es deportiva
            current_deporte = detectar_deporte(prompt)
            if not current_deporte:
                st.error("La consulta no parece estar relacionada con deportes. Por favor, formula una pregunta deportiva.")
                return
            
            # Indicar si el usuario requiere respuesta extendida
            detalle = any(p in prompt.lower() for p in PETICIONES_DETALLE)

            st.session_state.contexto.update({
                "deporte": current_deporte,
                "detalle": detalle,
                "nivel": st.session_state.get("user_prefs", {}).get("nivel", "básico")
            })

            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("🔍 Analizando..."):
                pass  # Aquí ya se tiene el deporte detectado
            
            with st.spinner("✍️ Generando respuesta..."):
                respuesta = generar_respuesta_mejorada(prompt, st.session_state.contexto)
                st.session_state.messages.append({"role": "assistant", "content": respuesta})
                st.rerun()

    with col2:
        if st.session_state.contexto.get("deporte"):
            mostrar_multimedia(st.session_state.contexto["deporte"])

        st.divider()
        st.markdown("#### 🗣️ ¿Te ayudó esta respuesta?")
        col_fb1, col_fb2 = st.columns(2)
        with col_fb1:
            if st.button("👍 Sí"):
                st.toast("¡Nos alegra que te sirviera! 🙌")
        with col_fb2:
            if st.button("👎 No mucho"):
                st.toast("Gracias por tu feedback. Mejoraremos. 💪")

if __name__ == "__main__":
    main()
