DESCRIPTION:
Ce script charge un modèle Random Forest pré-entraîné et effectue des prédictions
sur un fichier CSV de bornes de recharge (IRVE) pour la variable cible
"implantation_station".
 
 
PRÉREQUIS:
Bibliothèques Python nécessaires :
  - pandas, numpy, scikit-learn, joblib
 
Installer via : pip install pandas numpy scikit-learn joblib
 
Structure de dossiers attendue :
  ./Besoin_Client_3/
      ├── preprocessor.joblib
      ├── tuned_random_forest_model.pkl
      ├── ordinalencoder.joblib
      └── IRVE_clean_IA.csv
 
 
DONNÉES D'ENTRÉE:
Le CSV doit avoir subi exactement le même prétraitement R que le CSV
d'entraînement. Les colonnes requises sont :
 
  Booléennes   : paiement_acte, paiement_cb, paiement_autre,
                 reservation, cable_t2_attache
  Catégorielles: nom_operateur, restriction_gabarit, raccordement
  Numériques   : date_mise_en_service, horaires
 
 
UTILISATION:
1. Placez votre CSV prétraité dans ./Besoin_Client_3/IRVE_clean_IA.csv
   (ou modifiez la variable csv_path en fin de script).
 
2. Lancez le script :
     python predict.py
 
3. Le résultat est retourné sous forme de DataFrame avec une colonne
   "prediction" ajoutée. Les 10 premières lignes sont affichées en console.
 
 
MODIFIER LE FICHIER D'ENTRÉE:
Changez simplement la dernière ligne du script :
  csv_path = "./votre/chemin/fichier.csv"
 
 
SORTIES:
Le DataFrame enrichi (df_result) contient toutes les colonnes d'origine
plus la colonne "prediction" avec la valeur prédite d'implantation_station.
Pour sauvegarder le résultat :
  df_result.to_csv("resultats.csv", index=False)
 

EXEMPLE
-------
Un script d'exemple est fourni ci-dessous:
  import pandas as pd
  from besoin_client_3 import predict

  df = predict("./Besoin_Client_3/votre_csv.csv")
