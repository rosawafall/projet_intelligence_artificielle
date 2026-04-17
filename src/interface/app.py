import os
import sys
import json
import time
import tempfile
import glob
import random
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

# on remonte d'un niveau pour pouvoir importer src/agent/agent_llm
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.agent.agent_llm import run_agent, yolo_results_to_detections

# ---------------------------------------------------------
# Config de la page Streamlit
# ---------------------------------------------------------

st.set_page_config(
    page_title="Analyseur Conduite de Nuit",
    page_icon="🌙",
    layout="wide",
)

# un peu de CSS pour le style nocturne
st.markdown("""
<style>
    .titre { font-size:2rem; font-weight:800; color:#3498db; text-align:center; }
    .sous-titre { text-align:center; color:#7f8c8d; margin-bottom:2rem; }
    .risque-critique { background:#e74c3c; color:white; padding:6px 16px; border-radius:6px; font-weight:bold; display:inline-block; }
    .risque-eleve    { background:#e67e22; color:white; padding:6px 16px; border-radius:6px; font-weight:bold; display:inline-block; }
    .risque-moyen    { background:#f39c12; color:white; padding:6px 16px; border-radius:6px; font-weight:bold; display:inline-block; }
    .risque-faible   { background:#27ae60; color:white; padding:6px 16px; border-radius:6px; font-weight:bold; display:inline-block; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="titre">🌙 Analyseur de Scènes — Conduite de Nuit</div>', unsafe_allow_html=True)
st.markdown('<div class="sous-titre">YOLOv8 + Agent LLM Groq · Projet IA 2025–2026</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar : paramètres
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Configuration")

    groq_key = st.text_input("Clé Groq API", type="password",
                              value=os.getenv("GROQ_API_KEY", ""),
                              help="Gratuit sur console.groq.com")

    owm_key  = st.text_input("Clé OpenWeather (optionnel)", type="password",
                              value=os.getenv("OPENWEATHER_API_KEY", ""))

    ville    = st.text_input("Ville", value="Paris")

    st.divider()
    st.subheader("Détection YOLO")

    modele_choisi = st.selectbox("Modèle", ["YOLOv8s (recommandé)", "YOLOv8n (rapide)"])
    seuil_conf    = st.slider("Seuil de confiance", 0.10, 0.90, 0.30, 0.05,
                               help="On recommande 0.25–0.35 pour la nuit")

    st.divider()
    st.markdown("""
**Scénario : Conduite de Nuit**

Défis principaux :
- Piétons peu ou pas visibles
- Éblouissement par les phares
- Contraste très réduit
- Fatigue du conducteur

Classes détectées : piéton, voiture, camion, bus, feu, panneau
    """)


# ---------------------------------------------------------
# Chargement du modèle YOLO
# (mis en cache pour ne pas recharger à chaque interaction)
# ---------------------------------------------------------

@st.cache_resource(show_spinner="Chargement du modèle YOLO...")
def charger_modele(nom_modele):
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, "ultralytics non installé"

    # on cherche d'abord le modèle fine-tuné, sinon on prend le pretrained
    racine = Path(__file__).parent.parent.parent
    if "YOLOv8n" in nom_modele:
        chemin_finetuned = racine / "runs/train/conduite_nuit_yolov8n/weights/best.pt"
        chemin_base      = "yolov8n.pt"
    else:
        chemin_finetuned = racine / "runs/train/conduite_nuit_yolov8s/weights/best.pt"
        chemin_base      = "yolov8s.pt"

    if chemin_finetuned.exists():
        chemin = str(chemin_finetuned)
        info   = f"Modèle fine-tuné : {chemin_finetuned.name}"
    else:
        chemin = chemin_base
        info   = f"Modèle de base : {chemin_base} (pas encore fine-tuné)"

    modele = YOLO(chemin)
    return modele, info


# ---------------------------------------------------------
# Couleurs par classe pour l'affichage des bounding boxes
# ---------------------------------------------------------

COULEURS = {
    "pedestrian":    (231, 76,  60),
    "car":           (52,  152, 219),
    "truck":         (46,  204, 113),
    "bus":           (243, 156, 18),
    "traffic sign":  (155, 89,  182),
    "traffic light": (26,  188, 156),
}


def dessiner_boites(image, result_yolo, seuil):
    """Dessine les bounding boxes sur l'image avec la classe et le score."""
    img  = image.copy()
    draw = ImageDraw.Draw(img)

    for boite in result_yolo.boxes:
        conf = float(boite.conf[0])
        if conf < seuil:
            continue

        idx    = int(boite.cls[0])
        classe = result_yolo.names[idx]
        couleur = COULEURS.get(classe, (255, 255, 255))

        x1, y1, x2, y2 = [int(v) for v in boite.xyxy[0].tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=couleur, width=3)

        texte = f"{classe} {conf:.0%}"
        zone  = draw.textbbox((x1, y1 - 20), texte)
        draw.rectangle(zone, fill=couleur)
        draw.text((x1, y1 - 20), texte, fill="white")

    return img


def extraire_niveau_risque(rapport):
    """Cherche le niveau de risque dans le texte du rapport."""
    rapport_maj = rapport.upper()
    for niveau in ["CRITIQUE", "ÉLEVÉ", "ELEVE", "MOYEN", "FAIBLE"]:
        if niveau in rapport_maj:
            return "ÉLEVÉ" if niveau == "ELEVE" else niveau
    return "INCONNU"


# ---------------------------------------------------------
# Interface principale : 2 colonnes
# ---------------------------------------------------------

col_gauche, col_droite = st.columns([1, 1], gap="large")

with col_gauche:
    st.subheader("📸 Image dashcam")

    fichier = st.file_uploader("Chargez une image nocturne (JPG/PNG)", type=["jpg", "jpeg", "png"])

    st.caption("Pas d'image ? Utilisez une image de démo :")
    btn_demo = st.button("🎲 Image de démonstration")

    image = None

    if fichier is not None:
        image = Image.open(fichier).convert("RGB")
        st.image(image, caption="Image chargée", use_container_width=True)

    elif btn_demo:
        racine       = Path(__file__).parent.parent.parent
        dossier_demo = racine / "data/conduite_nuit/sample_images"
        images_demo  = list(dossier_demo.glob("*.jpg")) + list(dossier_demo.glob("*.png"))

        if images_demo:
            choix = random.choice(images_demo)
            image = Image.open(choix).convert("RGB")
            st.image(image, caption=f"Démo : {choix.name}", use_container_width=True)
        else:
            st.info("Placez des images JPG dans data/conduite_nuit/sample_images/")


with col_droite:
    st.subheader("🔍 Résultats")

    if image is None:
        # panneau d'accueil avec exemple
        st.info("⬅️ Chargez une image pour démarrer.")
        st.code(
            "ANALYSE DE SCÈNE — CONDUITE DE NUIT\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CONTEXTE : Route urbaine, nuit\n"
            "OBJETS DÉTECTÉS :\n"
            "  • 1 piéton (conf: 72%)\n"
            "  • 1 voiture (conf: 89%)\n"
            "DISTANCES ESTIMÉES :\n"
            "  • Piéton : ~6m → DANGER IMMÉDIAT\n"
            "NIVEAU DE RISQUE : ÉLEVÉ\n"
            "ANALYSE : ...\n"
            "RECOMMANDATIONS : ...",
            language="text"
        )

    else:
        modele, info_modele = charger_modele(modele_choisi)
        st.caption(info_modele)

        btn_analyser = st.button("🚀 Analyser", use_container_width=True)

        if btn_analyser:
            if not groq_key:
                st.error("Entrez votre clé Groq API dans la barre latérale.")
                st.stop()

            # --- Étape 1 : détection YOLO ---
            with st.spinner("Détection en cours..."):
                t0 = time.time()

                if modele is not None:
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        image.save(tmp.name)
                        resultats   = modele.predict(tmp.name, conf=seuil_conf, verbose=False)
                        result_yolo = resultats[0]
                        detections  = yolo_results_to_detections(result_yolo)
                        img_annotee = dessiner_boites(image, result_yolo, seuil_conf)
                        os.unlink(tmp.name)
                else:
                    # YOLO non disponible : on simule des détections
                    st.warning("YOLO non disponible, détections simulées.")
                    detections = [
                        {"class_name": "pedestrian",    "confidence": 0.72,
                         "bbox": {"x1":0.70,"y1":0.45,"x2":0.76,"y2":0.80}, "bbox_height":0.35},
                        {"class_name": "car",           "confidence": 0.89,
                         "bbox": {"x1":0.05,"y1":0.50,"x2":0.30,"y2":0.72}, "bbox_height":0.22},
                        {"class_name": "traffic light", "confidence": 0.81,
                         "bbox": {"x1":0.48,"y1":0.20,"x2":0.54,"y2":0.38}, "bbox_height":0.18},
                    ]
                    img_annotee = image.copy()

                temps_yolo = time.time() - t0

            st.image(img_annotee, caption="Détections YOLO", use_container_width=True)

            # comptage des objets par classe
            if detections:
                nb_par_classe = {}
                for d in detections:
                    cls = d["class_name"]
                    nb_par_classe[cls] = nb_par_classe.get(cls, 0) + 1

                colonnes = st.columns(min(len(nb_par_classe), 4))
                for i, (cls, nb) in enumerate(nb_par_classe.items()):
                    colonnes[i % len(colonnes)].metric(cls, nb)
            else:
                st.info("Aucun objet détecté — essayez un seuil plus bas.")

            # --- Étape 2 : analyse par l'agent LLM ---
            with st.spinner("Analyse par l'agent LLM Groq..."):
                t1 = time.time()

                os.environ["GROQ_API_KEY"] = groq_key
                if owm_key:
                    os.environ["OPENWEATHER_API_KEY"] = owm_key

                try:
                    rapport = run_agent(detections, city=ville)
                except Exception as e:
                    rapport = f"Erreur agent : {e}\n\nVérifiez votre clé Groq API."

                temps_agent = time.time() - t1

            # --- Affichage du rapport ---
            st.divider()
            niveau = extraire_niveau_risque(rapport)
            css_classe = {
                "CRITIQUE": "risque-critique",
                "ÉLEVÉ":    "risque-eleve",
                "MOYEN":    "risque-moyen",
                "FAIBLE":   "risque-faible",
            }.get(niveau, "risque-moyen")

            st.markdown(
                f'<span class="{css_classe}">⚠️ Niveau de risque : {niveau}</span>',
                unsafe_allow_html=True
            )
            st.markdown("")
            st.text_area("Rapport de l'agent", rapport, height=380)

            # métriques de temps
            temps_total = temps_yolo + temps_agent
            c1, c2, c3  = st.columns(3)
            c1.metric("Détection YOLO", f"{temps_yolo:.1f}s")
            c2.metric("Agent LLM",      f"{temps_agent:.1f}s")
            c3.metric("Total",          f"{temps_total:.1f}s",
                      delta="OK" if temps_total < 30 else "Lent")

            # export JSON du rapport
            export = {
                "scenario":   "Conduite de Nuit",
                "ville":      ville,
                "modele":     modele_choisi,
                "seuil_conf": seuil_conf,
                "detections": detections,
                "risque":     niveau,
                "rapport":    rapport,
            }
            st.download_button(
                "💾 Télécharger le rapport (JSON)",
                data=json.dumps(export, ensure_ascii=False, indent=2),
                file_name="rapport_conduite_nuit.json",
                mime="application/json",
            )

# pied de page
st.divider()
st.caption("Projet IA 2025–2026 · YOLOv8 + Groq LLaMA 3.3 70B + Streamlit · Scénario : Conduite de Nuit")
