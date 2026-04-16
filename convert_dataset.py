import json
import os
import random
import shutil

# les 6 classes qu'on garde pour notre scénario
CLASS_MAP = {
    'pedestrian':    0,
    'car':           1,
    'truck':         2,
    'bus':           3,
    'traffic sign':  4,
    'traffic light': 5,
}

# dimensions des images BDD100K
IMG_W = 1280
IMG_H = 720


def bbox_to_yolo(x1, y1, x2, y2):
    """
    Convertit une bounding box absolue en format YOLO normalisé.
    YOLO attend : x_centre, y_centre, largeur, hauteur — tous entre 0 et 1.
    """
    x_centre = ((x1 + x2) / 2) / IMG_W
    y_centre = ((y1 + y2) / 2) / IMG_H
    largeur  = (x2 - x1) / IMG_W
    hauteur  = (y2 - y1) / IMG_H
    return x_centre, y_centre, largeur, hauteur


def convertir_dataset(json_path, dossier_images, dossier_sortie, max_images=500, ratio_val=0.2):
    """
    Lit le JSON BDD100K, filtre les images de nuit et les convertit au format YOLO.

    Paramètres :
        json_path      : chemin vers le fichier JSON BDD100K (labels)
        dossier_images : dossier contenant les images BDD100K train/
        dossier_sortie : où on écrit les images + labels YOLO (data/conduite_nuit/)
        max_images     : combien d'images on prend maximum
        ratio_val      : proportion réservée à la validation (0.2 = 20%)
    """
    random.seed(42)

    # on crée les dossiers de sortie si besoin
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dossier_sortie, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dossier_sortie, 'labels', split), exist_ok=True)
    os.makedirs(os.path.join(dossier_sortie, 'sample_images'), exist_ok=True)

    # chargement du JSON
    print(f"Chargement du fichier JSON : {json_path}")
    with open(json_path, 'r') as f:
        donnees = json.load(f)
    print(f"Total d'images dans BDD100K : {len(donnees)}")

    # --- Filtrage : on garde seulement les images nocturnes ---
    images_nuit = []
    for entree in donnees:
        attributs = entree.get('attributes', {})

        # on vérifie l'heure de la journée
        heure = attributs.get('timeofday', '')
        if heure not in ['night', 'dawn/dusk']:
            continue

        # on vérifie qu'il y a au moins un objet qui nous intéresse
        categories = [label['category'] for label in entree.get('labels', [])]
        if not any(cat in CLASS_MAP for cat in categories):
            continue

        # on vérifie que l'image existe vraiment sur le disque
        chemin_image = os.path.join(dossier_images, entree['name'])
        if not os.path.exists(chemin_image):
            continue

        images_nuit.append(entree)

    print(f"Images nocturnes trouvées : {len(images_nuit)}")

    # on mélange et on prend les max_images premières
    random.shuffle(images_nuit)
    selection = images_nuit[:max_images]

    # séparation train / val
    nb_val   = int(len(selection) * ratio_val)
    noms_val = set(e['name'] for e in selection[:nb_val])

    compteur = {'train': 0, 'val': 0}
    nb_demo  = 0

    for entree in selection:
        nom_image = entree['name']
        split     = 'val' if nom_image in noms_val else 'train'
        src       = os.path.join(dossier_images, nom_image)

        # copie de l'image
        dest_img = os.path.join(dossier_sortie, 'images', split, nom_image)
        shutil.copy(src, dest_img)

        # génération du fichier label YOLO (.txt)
        nom_label   = nom_image.replace('.jpg', '.txt')
        lignes_yolo = []

        for label in entree.get('labels', []):
            cat = label['category']
            if cat not in CLASS_MAP:
                continue
            if 'box2d' not in label:
                continue

            b  = label['box2d']
            # on clamp les valeurs pour rester dans l'image
            x1 = max(0, min(IMG_W, b['x1']))
            y1 = max(0, min(IMG_H, b['y1']))
            x2 = max(0, min(IMG_W, b['x2']))
            y2 = max(0, min(IMG_H, b['y2']))

            if x2 <= x1 or y2 <= y1:
                continue  # bounding box invalide, on passe

            xc, yc, w, h = bbox_to_yolo(x1, y1, x2, y2)
            lignes_yolo.append(f"{CLASS_MAP[cat]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        dest_label = os.path.join(dossier_sortie, 'labels', split, nom_label)
        with open(dest_label, 'w') as f:
            f.write('\n'.join(lignes_yolo))

        compteur[split] += 1

        # on copie quelques images pour la démo Streamlit
        if split == 'train' and nb_demo < 5:
            shutil.copy(src, os.path.join(dossier_sortie, 'sample_images', nom_image))
            nb_demo += 1

    print(f"\nConversion terminée !")
    print(f"  Train : {compteur['train']} images")
    print(f"  Val   : {compteur['val']} images")

    # génération automatique du data.yaml pour YOLO
    yaml = f"""path: {os.path.abspath(dossier_sortie)}
train: images/train
val: images/val

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
"""
    chemin_yaml = os.path.join(os.path.dirname(dossier_sortie), 'data.yaml')
    with open(chemin_yaml, 'w') as f:
        f.write(yaml)
    print(f"  data.yaml généré : {chemin_yaml}")


if __name__ == "__main__":
    convertir_dataset(
        json_path      = 'data/labels/bdd100k_labels_images_train.json',
        dossier_images = 'data/images/train',
        dossier_sortie = 'data/conduite_nuit',
        max_images     = 500,
        ratio_val      = 0.2,
    )
