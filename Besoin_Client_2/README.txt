# PROJET IA - Besoin client 2 - Clustering

## 1. Description du programme
Ce module permet de prédire à quelle macro-région (cluster géographique de 0 à 4) appartient une borne de recharge électrique de la base de données IRVE.
Le script utilise un modèle d'IA (K-Means) préalablement entraîné sur les données. Il ne nécessite aucun réentraînement, ce qui garantit une exécution rapide (quelques secondes).

## 2. Prérequis et Architecture
Pour que le script fonctionne, les trois fichiers suivants doivent impérativement se trouver dans le même dossier :
* script_cluster.py : Le programme principal en Python.
* modele_kmeans.pkl : Le modèle K-Means entraîné sur la France hexagonale (K=5).
* scaler_gps.pkl : L'outil de normalisation des coordonnées géographiques.

**Bibliothèques Python requises :**
Assurez-vous que votre environnement Python dispose des bibliothèques suivantes :
`pip install numpy scikit-learn joblib`

## 3. Utilisation en ligne de commande (CLI)
Ouvrez un terminal, placez-vous dans le dossier contenant les fichiers, et exécutez le script en utilisant la syntaxe suivante :

Sur Mac/Linux :
> python3 script_cluster.py -lat <LATITUDE> -lon <LONGITUDE>

Sur Windows (si la commande ci-dessus échoue) :
> py script_cluster.py -lat <LATITUDE> -lon <LONGITUDE>

Arguments :
* `-lat` ou `--latitude` : La latitude de la borne au format décimal.
* `-lon` ou `--longitude` : La longitude de la borne au format décimal.

## 4. Exemple d'exécution
Pour tester une coordonnée située en plein centre-ville de Nantes (Latitude: 47.2184, Longitude: -1.5536) :

> py script_cluster.py -lat 47.2184 -lon -1.5536

Résultat attendu en console :
Cluster : 2



## 5. Notes techniques
* Le temps d'exécution de quelques secondes lors de l'appel du script correspond au chargement en mémoire vive des bibliothèques (scikit-learn) et des fichiers sérialisés (.pkl). L'inférence mathématique en elle-même est quasi-instantanée.
* Les coordonnées saisies doivent impérativement être séparées par un point (format anglo-saxon) et non une virgule.