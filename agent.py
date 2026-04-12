from groq import Groq
import requests

client = Groq(api_key="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")

def get_weather(city: str) -> str:
    r = requests.get(f"https://wttr.in/{city}?format=3")
    return r.text

def analyser_scene(detections: dict, ville: str = "Paris") -> str:
    meteo = get_weather(ville)
    prompt = f"""Tu es un expert en sécurité routière spécialisé dans la conduite de nuit.

Détections dans la scène : {detections}
Météo actuelle à {ville} : {meteo}

En tenant compte de la faible visibilité nocturne, génère un rapport :
- Niveau de risque : [FAIBLE/MOYEN/ÉLEVÉ/CRITIQUE]
- Éléments détectés : [liste]
- Analyse : [2-3 phrases tenant compte de la nuit et météo]
- Recommandations : [conseils conduite de nuit]"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    detections_test = {
        "voitures": [{"confiance": 0.92, "position": "proche"},
                     {"confiance": 0.87, "position": "milieu"}],
        "piétons": [{"confiance": 0.85, "position": "trottoir droit"}],
        "feux": "rouge",
        "panneaux": ["stop"]
    }
    rapport = analyser_scene(detections_test, "Paris")
    print("\n=== RAPPORT AGENT ===")
    print(rapport)