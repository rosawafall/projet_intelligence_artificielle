import os
import sys
import json
import time
import tempfile
import random
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
import requests
from groq import Groq

st.set_page_config(
    page_title="Analyseur Conduite de Nuit",
    page_icon="🌙",
    layout="wide",
)

st.markdown("""
<style>
.titre { font-size:2rem; font-weight:800; color:#3498db; text-align:center; }
.sous-titre { text-align:center; color:#7f8c8d; margin-bottom:2rem; }
.risque-critique { background:#e74c3c; color:white; padding:6px 16px; border-radius:6px; font-weight:bold; display:inline-block; }
.risque-eleve { background:#e67e22; color:white; padding:6px 16px; border-radius:6px; font-weight:bold; display:inline-block; }
.risque-moyen { background:#f39c12; color:white; padding:6px 16px; border-radius:6px; font-weight:bold; display:inline-block; }
.risque-faible { background:#27ae60; color:white; padding:6px 16px; border-radius:6px; font-weight:bold; display:inline-block; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="titre">🌙 Analyseur de Scènes — Conduite de Nuit</div>', unsafe_allow_html=True)
st.markdown('<div class="sous-titre">YOLOv8 + Agent LLM Groq · Projet IA 2025–2026</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input("Clé Groq API", type="password", value=os.getenv("GROQ_API_KEY", ""), help="Gratuit sur console.groq.com")
    ville = st.text_input("Ville", value="Paris")
    st.divider()
    st.subheader("Détection YOLO")
    modele_choisi = st.selectbox("Modèle", ["YOLOv8s (recommandé)", "YOLOv8n (rapide)"])
    seuil_conf = st.slider("Seuil de confiance", 0.10, 0.90, 0.30, 0.05)
    st.divider()
    st.markdown("""
**Scénario : Conduite de Nuit**

Défis principaux :
- Piétons peu ou pas visibles
- Éblouissement par les phares
- Contraste très réduit
- Fatigue du conducteur
""")

@st.cache_resource(show_spinner="Chargement du modèle YOLO...")
def charger_modele(nom_modele):
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, "ultralytics non installé"
    if "YOLOv8n" in nom_modele:
        chemin_base = "yolov8n.pt"
    else:
        chemin_base = "yolov8s.pt"
    info = f"Modèle de base : {chemin_base}"
    modele = YOLO(chemin_base)
    return modele, info

COULEURS = {
    "pedestrian": (231, 76, 60),
    "person": (231, 76, 60),
    "car": (52, 152, 219),
    "truck": (46, 204, 113),
    "bus": (243, 156, 18),
    "traffic sign": (155, 89, 182),
    "traffic light": (26, 188, 156),
}

def dessiner_boites(image, result_yolo, seuil):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for boite in result_yolo.boxes:
        conf = float(boite.conf[0])
        if conf < seuil:
            continue
        idx = int(boite.cls[0])
        classe = result_yolo.names[idx]
        couleur = COULEURS.get(classe, (255, 255, 255))
        x1, y1, x2, y2 = [int(v) for v in boite.xyxy[0].tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=couleur, width=3)
        texte = f"{classe} {conf:.0%}"
        try:
            zone = draw.textbbox((x1, max(y1 - 20, 0)), texte)
            draw.rectangle(zone, fill=couleur)
            draw.text((x1, max(y1 - 20, 0)), texte, fill="white")
        except:
            pass
    return img

def get_weather(city):
    try:
        r = requests.get(f"https://wttr.in/{city}?format=3", timeout=5)
        return r.text
    except:
        return f"Météo indisponible pour {city}"

def analyser_scene(detections, ville, groq_key):
    client = Groq(api_key=groq_key)
    meteo = get_weather(ville)
    prompt = f"""Tu es un expert en sécurité routière spécialisé dans la conduite de nuit.

Détections dans la scène : {json.dumps(detections, ensure_ascii=False)}
Météo actuelle à {ville} : {meteo}

En tenant compte de la faible visibilité nocturne, génère un rapport structuré :
- Niveau de risque : [FAIBLE/MOYEN/ÉLEVÉ/CRITIQUE]
- Éléments détectés : [liste avec scores de confiance]
- Analyse : [2-3 phrases tenant compte de la nuit et de la météo]
- Recommandations : [liste de conseils adaptés à la conduite de nuit]"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def extraire_niveau_risque(rapport):
    rapport_maj = rapport.upper()
    for niveau in ["CRITIQUE", "ÉLEVÉ", "ELEVE", "MOYEN", "FAIBLE"]:
        if niveau in rapport_maj:
            return "ÉLEVÉ" if niveau == "ELEVE" else niveau
    return "INCONNU"

col_gauche, col_droite = st.columns([1, 1], gap="large")

with col_gauche:
    st.subheader("📸 Image dashcam")
    fichier = st.file_uploader("Chargez une image nocturne (JPG/PNG)", type=["jpg", "jpeg", "png"])
    image = None
    if fichier is not None:
        image = Image.open(fichier).convert("RGB")
        st.image(image, caption="Image chargée", use_container_width=True)

with col_droite:
    st.subheader("🔍 Résultats")
    if image is None:
        st.info("⬅️ Chargez une image pour démarrer.")
    else:
        modele, info_modele = charger_modele(modele_choisi)
        st.caption(info_modele)
        btn_analyser = st.button("🚀 Analyser", use_container_width=True)
        if btn_analyser:
            if not groq_key:
                st.error("Entrez votre clé Groq API dans la barre latérale.")
                st.stop()

            with st.spinner("Détection YOLO en cours..."):
                t0 = time.time()
                if modele is not None:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        image.save(tmp.name)
                        tmp_name = tmp.name
                    resultats = modele.predict(tmp_name, conf=seuil_conf, verbose=False)
                    result_yolo = resultats[0]
                    img_annotee = dessiner_boites(image, result_yolo, seuil_conf)
                    detections = {}
                    for boite in result_yolo.boxes:
                        cls = result_yolo.names[int(boite.cls[0])]
                        conf = round(float(boite.conf[0]), 2)
                        if cls not in detections:
                            detections[cls] = []
                        detections[cls].append(conf)
                    try:
                        os.unlink(tmp_name)
                    except:
                        pass
                else:
                    st.warning("YOLO non disponible.")
                    detections = {"car": [0.89], "pedestrian": [0.72]}
                    img_annotee = image.copy()
                temps_yolo = time.time() - t0

            st.image(img_annotee, caption="Détections YOLO", use_container_width=True)

            if detections:
                colonnes = st.columns(min(len(detections), 4))
                for i, (cls, confs) in enumerate(detections.items()):
                    colonnes[i % len(colonnes)].metric(cls, len(confs))
            else:
                st.info("Aucun objet détecté — essayez un seuil plus bas.")

            with st.spinner("Analyse par l'agent LLM Groq..."):
                t1 = time.time()
                try:
                    rapport = analyser_scene(detections, ville, groq_key)
                except Exception as e:
                    rapport = f"Erreur agent : {e}"
                temps_agent = time.time() - t1

            st.divider()
            niveau = extraire_niveau_risque(rapport)
            css_classe = {
                "CRITIQUE": "risque-critique",
                "ÉLEVÉ": "risque-eleve",
                "MOYEN": "risque-moyen",
                "FAIBLE": "risque-faible",
            }.get(niveau, "risque-moyen")
            st.markdown(f'<span class="{css_classe}">⚠️ Niveau de risque : {niveau}</span>', unsafe_allow_html=True)
            st.markdown("")
            st.text_area("Rapport de l'agent", rapport, height=380)

            temps_total = temps_yolo + temps_agent
            c1, c2, c3 = st.columns(3)
            c1.metric("Détection YOLO", f"{temps_yolo:.1f}s")
            c2.metric("Agent LLM", f"{temps_agent:.1f}s")
            c3.metric("Total", f"{temps_total:.1f}s", delta="OK" if temps_total < 30 else "Lent")

            export = {
                "scenario": "Conduite de Nuit",
                "ville": ville,
                "modele": modele_choisi,
                "seuil_conf": seuil_conf,
                "detections": detections,
                "risque": niveau,
                "rapport": rapport,
            }
            st.download_button(
                "💾 Télécharger le rapport (JSON)",
                data=json.dumps(export, ensure_ascii=False, indent=2),
                file_name="rapport_conduite_nuit.json",
                mime="application/json",
            )

st.divider()
st.caption("Projet IA 2025–2026 · YOLOv8 + Groq LLaMA 3.3 70B + Streamlit · Scénario : Conduite de Nuit")