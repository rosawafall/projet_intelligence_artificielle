#  Analyseur Intelligent de Scènes de Conduite - Conduite de Nuit

**Projet de Fin de Module · Intelligence Artificielle · 2025–2026**
**Membres du groupe :** FALL Awa, Faye Rokhaya Ndambao et Anouar Zegguir.

Ce projet propose un pipeline complet d'IA pour assister la conduite nocturne : **Détection d'objets (YOLOv8) → Analyse par Agent LLM (Groq) → Rapport de sécurité en temps réel**.

---

## Structure du Projet

Le code est organisé pour séparer la préparation des données, l'entraînement et l'interface utilisateur :

### 1. Notebooks (Google Colab)
* `01_exploration_et_entrainement.ipynb` : Exploration du dataset BDD100K, filtrage des scènes de nuit, conversion au format YOLO et entraînement comparatif des modèles.
* `02_demo_agent.ipynb` : Démonstration isolée de l'agent LLM Groq avec ses outils (météo, distance, code de la route).

### 2. Code Source (VS Code / Local)
* **`src/detection/convert_dataset.py`** : Script pour transformer les labels JSON BDD100K en fichiers `.txt` compatibles YOLO.
* **`src/agent/agent_llm.py`** : Cœur de l'intelligence avec l'implémentation de l'agent Groq (Llama 3.3 70B) et ses 3 fonctions d'outils.
* **`src/interface/app.py`** : Interface web interactive développée avec **Streamlit**.

### 3. Configuration & Données
* `data/data.yaml` : Fichier de configuration YOLO définissant les classes (piéton, voiture, camion, etc.).
* `requirement.txt` : Liste des dépendances (ultralytics, groq, streamlit, etc.).

---

##  Scénario et Détection (YOLOv8)

Nous avons choisi le scénario de **Conduite de Nuit** pour répondre aux défis de visibilité réduite.

### Modèles Testés : YOLOv8n vs YOLOv8s
Nous avons entraîné et comparé deux versions du modèle YOLO sur le dataset **BDD100K** :
* **YOLOv8n (Nano)** : Plus rapide, idéal pour l'inférence en temps réel sur des appareils légers.
* **YOLOv8s (Small)** : Un peu plus lourd mais offre une meilleure précision (mAP) pour détecter les petits objets dans l'obscurité.

---

##  L'Agent IA et ses Outils

L'agent utilise l'API **Groq** pour raisonner sur les objets détectés par YOLO. Il dispose de 3 outils clés :
1.  **Météo (OpenWeatherMap)** : Pour ajuster le risque selon la visibilité (pluie, brouillard).
2.  **Estimation de Distance** : Calcule la proximité des dangers selon la taille des "bounding boxes".
3.  **Règles de Conduite de Nuit** : Fournit des recommandations basées sur le code de la route.

---

## Installation et Exécution

### 1. Installation des dépendances
```bash
pip install -r requirement.txt
```

### 2. Configuration des clés API
Créez un fichier `.env` ou exportez vos clés :
```bash
export GROQ_API_KEY="votre_cle_groq"
```

### 3. Lancer l'interface Streamlit
Pour tester le pipeline complet avec une interface graphique :
```bash
streamlit run src/interface/app.py
```

### 4. Tests sur Colab
Pour l'entraînement ou la démo de l'agent sans installation locale, ouvrez les fichiers `.ipynb` dans **Google Colab** et exécutez les cellules séquentiellement.
