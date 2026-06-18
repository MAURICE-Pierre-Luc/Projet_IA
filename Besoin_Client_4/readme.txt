BESOIN CLIENT 4 - PRÉDICTION DE LA PUISSANCE NOMINALE DES BORNES IRVE

Description :
Ce script utilise un modèle d'Intelligence Artificielle (Random Forest) 
pour prédire la puissance nominale d'une borne de recharge en fonction 
de ses caractéristiques techniques, commerciales et géographiques.

Contenu du dossier :
- IRVE_clean_IA.csv : Base de données nettoyée.
- IA_experimental.ipynb : Notebook d'entraînement et d'évaluation.
- best_classifier_besoin4.pkl : Modèle d'IA sauvegardé.
- preprocessor_besoin4.pkl : Pipeline d'encodage des variables textuelles.
- prediction_puissance.py : Script autonome de prédiction interactif.

Le projet est articulé autour de deux grandes étapes : l'entraînement du modèle et son utilisation interactive.

1. Entraînement et Optimisation (`train.py` ou Notebook)
1. Prétraitement des données : Le script applique le pipeline de transformation (One Hot Encoding pour les catégories textuelles, StandardScaler pour les valeurs numériques).
2. Gestion du déséquilibre des classes : La classe 22 kW étant ultra-majoritaire sur le territoire, l'algorithme est configuré avec le paramètre `class_weight='balanced'`. Cela force l'IA à accorder une importance équitable aux puissances plus rares (comme les bornes de charge rapide 50 kW, 150 kW ou 300 kW).
3. Sélection et Sauvegarde : Après une recherche par grille (GridSearch), le meilleur estimateur est automatiquement sauvegardé sous forme de fichier sérialisé : `best_classifier_besoin4.pckl`.

 2. Interface de Test et de Prédiction
Le script de prédiction interactif (interface de test utilisateur) charge le modèle entraîné pour évaluer une configuration à la volée :
a. Saisie utilisateur :L'utilisateur renseigne les caractéristiques de la borne à tester (types de prises disponibles, coordonnées géographiques, capacité de la station, écosystème commercial).
b. Alignement du pipeline :Les entrées saisies par l'utilisateur sont formatées de manière à correspondre exactement aux 2795 dimensions apprises lors de l'entraînement.
c. Inférence de l'IA :Le modèle calculé par le Random Forest évalue les probabilités et retourne la classe de puissance nominale estimée la plus cohérente (ex: `22.0 kW`, `50.0 kW`, `150.0 kW`).
