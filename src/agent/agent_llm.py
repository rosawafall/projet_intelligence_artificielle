import os
import json
import requests
from groq import Groq

# on récupère les clés API depuis les variables d'environnement
# (ou on met une valeur par défaut pour les tests)
GROQ_KEY = os.getenv("GROQ_API_KEY", "")
OWM_KEY  = os.getenv("OPENWEATHER_API_KEY", "")
MODELE   = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_KEY)


# -------------------------------------------------------
# OUTILS DE L'AGENT
# On définit 3 outils que l'agent peut appeler :
#   1. météo (OpenWeatherMap)
#   2. estimation de distance (depuis la taille de la bbox)
#   3. règles de conduite nocturne
# -------------------------------------------------------

OUTILS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Retourne la météo actuelle d'une ville. Utile pour savoir si la visibilité est réduite (pluie, brouillard).",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Nom de la ville, ex: 'Paris'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_distance",
            "description": "Estime la distance d'un objet détecté en mètres, à partir de la hauteur relative de sa bounding box.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_height_ratio": {
                        "type": "number",
                        "description": "Hauteur de la bbox divisée par la hauteur de l'image (valeur entre 0 et 1)"
                    },
                    "object_type": {
                        "type": "string",
                        "description": "Type d'objet : 'pedestrian', 'car', 'truck' ou 'bus'"
                    }
                },
                "required": ["bbox_height_ratio", "object_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_night_driving_rules",
            "description": "Retourne les règles du code de la route pour la conduite de nuit (vitesses, distances, feux...).",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Pays concerné, ex: 'France'"
                    }
                },
                "required": ["country"]
            }
        }
    }
]


# -------------------------------------------------------
# Implémentation des 3 outils
# -------------------------------------------------------

def get_weather(city):
    """Appelle l'API OpenWeatherMap pour récupérer la météo."""
    if not OWM_KEY or OWM_KEY == "YOUR_OWM_KEY":
        # pas de clé API → on simule une météo par défaut
        return {
            "city": city,
            "description": "ciel dégagé",
            "temperature": 12,
            "humidity": 65,
            "wind_speed": 4.1,
            "visibility_m": 9000,
            "note": "Données simulées (pas de clé OpenWeather)"
        }

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OWM_KEY}&units=metric&lang=fr"
        reponse = requests.get(url, timeout=5)
        data    = reponse.json()
        return {
            "city":         city,
            "description":  data["weather"][0]["description"],
            "temperature":  data["main"]["temp"],
            "humidity":     data["main"]["humidity"],
            "wind_speed":   data["wind"]["speed"],
            "visibility_m": data.get("visibility", 10000),
        }
    except Exception as e:
        return {"city": city, "erreur": str(e), "visibility_m": 9000}


def estimate_distance(bbox_height_ratio, object_type):
    """
    Estime la distance d'un objet en mètres.
    
    Principe : on connaît la taille réelle de l'objet (hauteur en mètres),
    et on mesure sa taille apparente dans l'image (bbox_height_ratio).
    Plus l'objet paraît petit dans l'image, plus il est loin.
    
    La nuit on ajoute +20% d'incertitude car la détection est moins précise.
    """
    # hauteur réelle approximative de chaque type d'objet
    hauteurs_reelles = {
        'pedestrian': 1.75,
        'car':        1.50,
        'truck':      3.50,
        'bus':        3.20,
    }

    if bbox_height_ratio <= 0.01:
        return {"distance_m": None, "proximite": "indéterminée"}

    h_reel   = hauteurs_reelles.get(object_type, 1.75)
    distance = (h_reel / bbox_height_ratio) * 0.9  # facteur de champ de vision
    distance_nuit = round(distance * 1.2, 1)        # +20% d'incertitude la nuit

    if distance_nuit < 8:
        proximite = "très proche — DANGER IMMÉDIAT"
    elif distance_nuit < 20:
        proximite = "proche — vigilance maximale"
    elif distance_nuit < 40:
        proximite = "distance moyenne"
    else:
        proximite = "éloigné"

    return {
        "distance_estimee_m": distance_nuit,
        "object_type":        object_type,
        "proximite":          proximite,
        "note":               "incertitude +20% (conditions nocturnes)"
    }


def get_night_driving_rules(country="France"):
    """Retourne les règles de conduite nocturne pour la France."""
    regles = {
        "France": {
            "vitesse_hors_agglomeration": 80,
            "vitesse_agglomeration": 50,
            "vitesse_mauvaise_visibilite": 50,
            "distance_securite_min_m": 15,
            "feux": "feux de croisement obligatoires",
            "alertes": [
                "Piéton sans vêtements réfléchissants",
                "Visibilité inférieure à 50m",
                "Éblouissement par phares adverses",
                "Somnolence du conducteur",
            ],
            "conseils": [
                "Réduire la vitesse de 20% par rapport à la limite de jour",
                "Allumer les feux de croisement dès le coucher du soleil",
                "Garder au moins 15m de distance avec le véhicule devant",
                "S'arrêter si la visibilité tombe sous 50 mètres",
            ]
        }
    }
    return regles.get(country, regles["France"])


def executer_outil(nom_outil, arguments):
    """Appelle le bon outil selon ce que l'agent a demandé."""
    if nom_outil == "get_weather":
        resultat = get_weather(**arguments)
    elif nom_outil == "estimate_distance":
        resultat = estimate_distance(**arguments)
    elif nom_outil == "get_night_driving_rules":
        resultat = get_night_driving_rules(**arguments)
    else:
        resultat = {"erreur": f"Outil inconnu : {nom_outil}"}
    return json.dumps(resultat, ensure_ascii=False)


# -------------------------------------------------------
# Prompt système : on explique à l'agent son rôle
# -------------------------------------------------------

SYSTEM_PROMPT = """Tu es un expert en sécurité routière spécialisé dans la conduite de nuit.

Tu reçois les résultats d'une détection YOLO sur une image dashcam nocturne.
Ton rôle est d'analyser la scène et de produire un rapport de sécurité structuré.

Pour chaque analyse tu dois :
1. Appeler get_night_driving_rules pour connaître les règles applicables
2. Appeler estimate_distance pour les objets détectés (piétons et véhicules surtout)
3. Appeler get_weather pour évaluer la visibilité météo

Format de réponse attendu :
ANALYSE DE SCÈNE — CONDUITE DE NUIT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXTE : [description de la scène]
OBJETS DÉTECTÉS : [liste avec scores de confiance]
DISTANCES ESTIMÉES : [distances calculées]
NIVEAU DE RISQUE : [FAIBLE / MOYEN / ÉLEVÉ / CRITIQUE]
ANALYSE : [explication en 2-3 phrases]
RECOMMANDATIONS :
  • [conseil 1]
  • [conseil 2]
  • [conseil 3]
CONDITIONS : [météo et visibilité]

La nuit, sois particulièrement attentif aux piétons qui sont difficiles à détecter."""


# -------------------------------------------------------
# Boucle principale de l'agent
# -------------------------------------------------------

def run_agent(detections, city="Paris"):
    """
    Lance l'agent LLM sur une liste de détections YOLO.
    
    L'agent va appeler ses outils (météo, distance, règles) puis
    synthétiser tout ça dans un rapport de sécurité.
    
    Paramètres :
        detections : liste de dicts (class_name, confidence, bbox, bbox_height)
        city       : ville pour la météo
    
    Retourne : le rapport texte de l'agent
    """
    # on prépare le message utilisateur avec les détections
    message_user = f"""Analyse cette scène de conduite nocturne.

Détections YOLO :
{json.dumps(detections, ensure_ascii=False, indent=2)}

Ville pour la météo : {city}

Génère un rapport complet adapté aux conditions de nuit."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": message_user},
    ]

    # boucle de l'agent : max 6 tours pour éviter les boucles infinies
    for _ in range(6):
        reponse = client.chat.completions.create(
            model       = MODELE,
            messages    = messages,
            tools       = OUTILS,
            tool_choice = "auto",
            max_tokens  = 1500,
            temperature = 0.3,
        )

        msg = reponse.choices[0].message

        # si l'agent n'a pas demandé d'outil → c'est la réponse finale
        if not msg.tool_calls:
            return msg.content

        # sinon on ajoute la réponse de l'agent à l'historique
        messages.append({
            "role":       "assistant",
            "content":    msg.content or "",
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in msg.tool_calls
            ],
        })

        # on exécute chaque outil demandé et on renvoie le résultat
        for tc in msg.tool_calls:
            args    = json.loads(tc.function.arguments)
            resulat = executer_outil(tc.function.name, args)
            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      resulat,
            })

    return "L'agent n'a pas pu produire de réponse après plusieurs essais."


# -------------------------------------------------------
# Utilitaire : convertit les résultats YOLO bruts
# en format lisible par run_agent()
# -------------------------------------------------------

def yolo_results_to_detections(yolo_result):
    """
    Transforme l'objet de résultat YOLO (Ultralytics) en une liste
    de dicts simples qu'on peut passer à run_agent().
    """
    detections = []
    noms   = yolo_result.names
    boites = yolo_result.boxes

    for boite in boites:
        idx        = int(boite.cls[0])
        classe     = noms[idx]
        confiance  = float(boite.conf[0])
        x1, y1, x2, y2 = boite.xyxyn[0].tolist()  # coordonnées normalisées

        detections.append({
            "class_name":  classe,
            "confidence":  round(confiance, 3),
            "bbox": {
                "x1": round(x1, 4),
                "y1": round(y1, 4),
                "x2": round(x2, 4),
                "y2": round(y2, 4),
            },
            "bbox_height": round(y2 - y1, 4),
        })

    return detections


# -------------------------------------------------------
# Test rapide si on lance ce fichier directement
# -------------------------------------------------------

if __name__ == "__main__":
    # exemple de détections pour tester sans YOLO
    exemple = [
        {
            "class_name":  "pedestrian",
            "confidence":  0.72,
            "bbox":        {"x1": 0.70, "y1": 0.45, "x2": 0.76, "y2": 0.80},
            "bbox_height": 0.35
        },
        {
            "class_name":  "car",
            "confidence":  0.89,
            "bbox":        {"x1": 0.05, "y1": 0.50, "x2": 0.30, "y2": 0.72},
            "bbox_height": 0.22
        },
        {
            "class_name":  "traffic light",
            "confidence":  0.81,
            "bbox":        {"x1": 0.48, "y1": 0.20, "x2": 0.54, "y2": 0.38},
            "bbox_height": 0.18
        },
    ]

    print("Lancement de l'agent...\n")
    rapport = run_agent(exemple, city="Paris")
    print(rapport)
