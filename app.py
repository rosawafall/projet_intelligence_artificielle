import os
import streamlit as st
import requests
import numpy as np
from PIL import Image
from groq import Groq
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = YOLO("yolov8n.pt")

st.set_page_config(
    page_title="Analyseur de conduite de nuit",
    page_icon="🌙",
    layout="wide"
)

def get_weather(city):
    r = requests.get(f"https://wttr.in/{city}?format=3")
    return r.text

def analyser_scene(detections, ville="Paris"):
    meteo = get_weather(ville)
    prompt = f"""Tu es un expert en sécurité routière spécialisé dans la conduite de nuit.
Détections : {detections}
Météo à {ville} : {meteo}
Génère un rapport :
- Niveau de risque : [FAIBLE/MOYEN/ÉLEVÉ/CRITIQUE]
- Éléments détectés : [liste]
- Analyse : [2-3 phrases tenant compte de la nuit]
- Recommandations : [conseils conduite de nuit]"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.title("🌙 Analyseur Intelligent de Scènes de Conduite de Nuit")
st.markdown("---")

with st.sidebar:
    st.header("Paramètres")
    ville = st.text_input("Ville", value="Paris")
    conf_threshold = st.slider("Seuil de confiance", 0.1, 0.9, 0.3)
    st.markdown("---")
    st.markdown("**Modèle :** YOLOv8n")
    st.markdown("**LLM :** Llama 3.3 70B")
    st.markdown("**Scénario :** Conduite de nuit")

uploaded = st.file_uploader("Upload une image dashcam nocturne", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    img_array = np.array(image)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(image, use_column_width=True)

    with st.spinner("Détection YOLOv8 en cours..."):
        results = model.predict(img_array, conf=conf_threshold)
        annotated = results[0].plot()
        detections = {}
        for box in results[0].boxes:
            cls = results[0].names[int(box.cls)]
            conf = float(box.conf)
            if cls not in detections:
                detections[cls] = []
            detections[cls].append(round(conf, 2))

    with col2:
        st.subheader("Détections YOLOv8")
        st.image(annotated, use_column_width=True)

    st.markdown("---")
    cols = st.columns(3)
    cols[0].metric("Objets détectés", sum(len(v) for v in detections.values()))
    cols[1].metric("Classes distinctes", len(detections))
    cols[2].metric("Ville", ville)

    st.markdown("---")
    with st.spinner("Analyse agent IA..."):
        rapport = analyser_scene(detections, ville)

    st.subheader("📋 Rapport de l'agent IA")
    st.markdown(rapport)
else:
    st.info("Upload une image dashcam nocturne pour commencer l'analyse.")