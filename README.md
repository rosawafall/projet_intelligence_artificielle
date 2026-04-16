# 🌙 Analyseur Intelligent de Scènes de Conduite — Conduite de Nuit

**Projet IA · Module Intelligence Artificielle · 2025–2026**

Pipeline complet : **Image dashcam → YOLOv8 (détection) → Agent LLM Groq → Rapport de risque**

---

## 📁 Structure du projet

```
PROJET_CONDUITE_NUIT/
│
├── data/
│   ├── data.yaml                          ← config YOLO (classes + chemins)
│   ├── images/                            ← nos images BDD100K 
│   │   ├── train/                         ← images train BDD100K brutes
│   │   └── val/                           ← images val BDD100K brutes 
│   ├── labels/                            ← labels JSON BDD100K
│   │   ├── train/                         ← labels train BDD100K brutes
│   │   └── val/
│   └── conduite_nuit/                     ← généré par convert_dataset.py
│       ├── images/
│       │   ├── train/                     ← images nocturnes filtrées (80%)
│       │   └── val/                       ← images nocturnes validation (20%)
│       ├── labels/
│       │   ├── train/                     ← annotations YOLO .txt
│       │   └── val/
│       └── sample_images/                 ← 5 images pour la démo Streamlit
│
├── notebooks/
│   ├── Exploration de Données.ipynb       ← stats + visualisations dataset
│   └── training/
│       └── training.ipynb                 ← entraînement + comparaison métriques
│
├── src/
│   ├── detection/
│   │   └── convert_dataset.py             ← filtre BDD100K → YOLO (nuit)
│   ├── agent/
│   │   └── agent_llm.py                   ← agent Groq LLaMA 3.3 + 3 outils
│   └── interface/
│       └── app.py                         ← interface Streamlit complète
│
├── runs/                                  ← généré par YOLO (entraînement)
│   ├── train/
│   │   ├── conduite_nuit_yolov8n/weights/best.pt
│   │   └── conduite_nuit_yolov8s/weights/best.pt
│   └── predict/
│
├── README.md
├── requirement.txt
└── .env                                   ← nos clés API
```

---

## ⚙️ Installation

```bash
pip install -r requirement.txt
```
---

## 🚀 Utilisation — 3 étapes

### Étape 1 — Convertir le dataset BDD100K

```bash
python src/detection/convert_dataset.py \
  --labels-dir data/labels \
  --images-dir data/images \
  --output-dir data/conduite_nuit \
  --max-images 500
```

**Résultat** : crée `data/conduite_nuit/` avec images + labels YOLO filtrés **nuit/aube/crépuscule**.

---

### Étape 2 — Entraîner les modèles YOLOv8

Ouvrir et exécuter `notebooks/training/training.ipynb`.

> 💡 **Google Colab (GPU T4 gratuit) fortement recommandé**

Le notebook entraîne et compare **YOLOv8n** vs **YOLOv8s** et génère :
- `runs/train/conduite_nuit_yolov8n/weights/best.pt`
- `runs/train/conduite_nuit_yolov8s/weights/best.pt`
- Graphiques de métriques dans `data/conduite_nuit/`

---

### Étape 3 — Lancer l'interface Streamlit

```bash
streamlit run src/interface/app.py
```

L'interface permet de :
- Uploader une image dashcam nocturne
- Voir les détections YOLO avec bounding boxes colorées
- Lire le rapport de l'agent LLM (risque + recommandations)
- Télécharger le rapport au format JSON


## 🌙 Scénario — Conduite de Nuit

| Élément | Détail |
|---|---|
| Dataset | BDD100K filtré `timeofday: night / dawn/dusk` |
| Modèles | YOLOv8n (léger) vs YOLOv8s (précis) — fine-tunés |
| Classes | `pedestrian`, `car`, `truck`, `bus`, `traffic sign`, `traffic light` |
| Seuil conf. | 0.30 (réduit pour capturer les piétons peu visibles) |
| Agent LLM | Groq LLaMA 3.3 70B + météo + distance + règles nocturnes |
| Défis | Visibilité réduite, éblouissement phares, piétons non réfléchissants |

---

## 🤖 Architecture de l'agent LLM

L'agent utilise le **tool calling** avec 3 outils :

| Outil | Description |
|---|---|
| `get_weather` | Météo actuelle via OpenWeatherMap (visibilité, pluie, brouillard) |
| `estimate_distance` | Distance objet via ratio bounding box (+20% d'incertitude la nuit) |
| `get_night_driving_rules` | Règles code de la route nocturne (France) |

**Boucle agent** : observation → appels outils → réponse structurée (5 tours max)

---

## 📊 Métriques attendues (BDD100K, 500 images nuit)

| Modèle | mAP@0.5 | mAP@0.5:0.95 | Précision | Rappel |
|---|---|---|---|---|
| YOLOv8n | ~0.45–0.55 | ~0.28–0.35 | ~0.65 | ~0.50 |
| YOLOv8s | ~0.52–0.62 | ~0.33–0.42 | ~0.70 | ~0.55 |

> ⚠️ Les métriques nocturnes sont inférieures aux métriques de jour — c'est attendu et constitue le défi principal du scénario.

---

## 📋 Exemple de rapport généré

```
ANALYSE DE SCÈNE — CONDUITE DE NUIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXTE : Route urbaine nocturne, faible éclairage public
OBJETS DÉTECTÉS :
   • 1 piéton    — conf: 72%, côté droit
   • 1 voiture   — conf: 89%, phares allumés, face avant
   • 1 feu rouge — conf: 81%, centre supérieur
DISTANCES ESTIMÉES :
   • Piéton  : ~6.3m → très proche — DANGER IMMÉDIAT
   • Voiture : ~21.6m → distance moyenne
NIVEAU DE RISQUE : ÉLEVÉ
ANALYSE : Piéton détecté à distance critique dans des conditions
de visibilité réduite. La nuit amplifie le risque de collision.
Feu rouge actif — arrêt obligatoire.
RECOMMANDATIONS :
  • Réduire immédiatement la vitesse à 30 km/h
  • Prioriser la sécurité du piéton — freinage anticipé
  • Maintenir 15m de distance de sécurité avec le véhicule
CONDITIONS : Ciel dégagé, 12°C, visibilité estimée 9 000m
```

---

## 🛠️ Dépannage

| Problème | Solution |
|---|---|
| `ModuleNotFoundError: groq` | `pip install groq` |
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| `best.pt not found` | Exécuter d'abord `training.ipynb` |
| Aucun objet détecté | Réduire le seuil de confiance à 0.20–0.25 |
| Erreur Groq API | Vérifier la clé dans la sidebar ou le fichier `.env` |
| Images BDD100K non trouvées | Vérifier `data/images/train/` et `data/labels/` |

---

## 📎 Ressources

- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Groq Console](https://console.groq.com/)
- [Streamlit Docs](https://docs.streamlit.io/)
