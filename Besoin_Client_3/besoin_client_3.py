import os
import re
import numpy as np
import pandas as pd
import joblib
from collections import defaultdict


#Il est très important que le csv ait subit le prétraitement R que le csv utilisé pour l'entrainement à subit

# Chemins vers les artefacts models, encoders et ordinal encoder sauvegardés lors de l'entraînement
MODELS_DIR = "./Besoin_Client_3/models"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "tuned_random_forest_model.pkl")
ORDINAL_ENCODER_PATH = os.path.join(MODELS_DIR, "ordinalencoder.joblib")


# Définition des colonnes par type, telles qu'utilisées à l'entraînement.
BOOL_COLS = [
    "paiement_acte", "paiement_cb", "paiement_autre",
    "reservation", "cable_t2_attache",
]

CAT_COLS = [
    "nom_operateur", "restriction_gabarit", "raccordement",
]

NUM_COLS = [
    "annee_mise_en_service",
    "horaires",
]

# Liste complète des features dans l'ordre attendu par le préprocesseur
FEATURE_COLS = BOOL_COLS + CAT_COLS + NUM_COLS


def load_model_and_preprocessor():
    """Charge le préprocesseur, l'OrdinalEncoder et le modèle enregistrés."""
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        ordinal_encoder = joblib.load(ORDINAL_ENCODER_PATH)
        model = joblib.load(MODEL_PATH)
        print("Préprocesseur, OrdinalEncoder et modèle chargés avec succès.")
        return preprocessor, ordinal_encoder, model
    except FileNotFoundError:
        print(
            "Erreur: Assurez-vous que les fichiers preprocessor.joblib, "
            "ordinalencoder.joblib et tuned_random_forest_model.pkl "
            "existent dans le dossier spécifié."
        )
        return None, None, None


# Parsing des horaires

DAYS_ORDER = ["mo", "tu", "we", "th", "fr", "sa", "su"]


def _expand_days(day_part: str) -> list:
    """Convertit une expression de jours (ex: 'mo-fr', 'sa,su') en liste de jours."""
    day_part = day_part.strip()
    if "," in day_part:
        result = []
        for d in day_part.split(","):
            result.extend(_expand_days(d.strip()))
        return result
    if "-" in day_part:
        start, end = day_part.split("-", 1)
        start, end = start.strip(), end.strip()
        if start not in DAYS_ORDER or end not in DAYS_ORDER:
            return []
        i1, i2 = DAYS_ORDER.index(start), DAYS_ORDER.index(end)
        if i1 <= i2:
            return DAYS_ORDER[i1 : i2 + 1]
        return DAYS_ORDER[i1:] + DAYS_ORDER[: i2 + 1]
    return [day_part] if day_part in DAYS_ORDER else []


def _parse_hours(h: str):
    """Extrait le tuple (heure_début, heure_fin) depuis une chaîne 'HH:MM-HH:MM'."""
    m = re.match(r"^\s*(\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})\s*$", h)
    if not m:
        return None
    return m.group(1), m.group(2)


def _parse_line(line: str) -> dict:
    """Parse une ligne d'horaires complète en un dict {jour: [(début, fin), ...]}."""
    if not isinstance(line, str):
        return {}
    result = defaultdict(list)
    line = line.replace(";", ",")
    line = re.sub(r"\s+", " ", line.strip())
    blocks = re.findall(r"([a-z, -]+)\s+(\d{2}:\d{2}\s*-\s*\d{2}:\d{2})", line)
    for days_raw, hours_raw in blocks:
        days = _expand_days(days_raw)
        hours = _parse_hours(hours_raw)
        if not days or not hours:
            continue
        start, end = hours
        for d in days:
            result[d].append((start, end))
    return dict(result)


def _time_to_minutes(t: str) -> int:
    """Convertit une heure 'HH:MM' en nombre de minutes depuis minuit."""
    h, m = map(int, t.split(":"))
    return h * 60 + m


def _weekly_hours(schedule: dict) -> float:
    """Calcule le total d'heures d'ouverture hebdomadaires depuis un planning parsé."""
    total_minutes = 0
    for periods in schedule.values():
        for start, end in periods:
            total_minutes += _time_to_minutes(end) - _time_to_minutes(start)
    return round(total_minutes / 60, 2)



#Prepocession des donnes du csv

def preprocess_new_data(
    df_input: pd.DataFrame,
    preprocessor,
    feature_cols: list = FEATURE_COLS,
    bool_cols: list = BOOL_COLS,
    cat_cols: list = CAT_COLS,
    num_cols: list = NUM_COLS,
) -> np.ndarray:
    """Prétraite les nouvelles données en utilisant le préprocesseur enregistré.

    Args:
        df_input: DataFrame contenant les nouvelles données.
        preprocessor: Le préprocesseur sklearn chargé.
        feature_cols: Liste des colonnes de features.
        bool_cols: Liste des colonnes booléennes.
        cat_cols: Liste des colonnes catégorielles.
        num_cols: Liste des colonnes numériques.

    Returns:
        np.ndarray: Données prétraitées prêtes pour la prédiction.
    """
    df_processed = df_input.copy()

    # 1. Traitement des booléens
    # Les valeurs texte ("true"/"false"/"inconnu") sont converties en 0/1/NaN
    def convertir_booleen(valeur):
        correspondances = {
            "true": 1.0, "false": 0.0, "inconnu": np.nan,
            True: 1.0, False: 0.0,
        }
        cle = str(valeur).strip().lower()
        return correspondances.get(cle, np.nan)

    for col in bool_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].map(convertir_booleen)

    # 2. Nettoyage des catégorielles
    # Les valeurs "inconnu" sont remplacées par NaN pour être gérées par l'imputation
    for col in cat_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].replace("inconnu", pd.NA)

    # 3. Extraction de l'année de mise en service
    # La date complète n'est pas utilisée directement : seule l'année est extraite
    if "date_mise_en_service" in df_processed.columns and "annee_mise_en_service" not in df_processed.columns:
        def extraire_annee(valeur):
            if pd.isna(valeur) or str(valeur).strip().lower() == "inconnu":
                return pd.NA
            try:
                return float(str(valeur).strip()[:4])
            except ValueError:
                return pd.NA

        df_processed["annee_mise_en_service"] = df_processed[
            "date_mise_en_service"
        ].map(extraire_annee)

    # 4. Conversion des horaires en nombre d'heures hebdomadaires
    if "horaires" in df_processed.columns and df_processed["horaires"].dtype == object:
        df_processed["horaires"] = (
            df_processed["horaires"]
            .map(_parse_line)
            .map(_weekly_hours)
        )

    # 5. Vérification des colonnes attendues
    missing_cols = [c for c in feature_cols if c not in df_processed.columns]
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans les données d'entrée : {missing_cols}"
        )

    X_input = df_processed[feature_cols].copy()

    # Remplacer les -1 (encodage ordinal des inconnus) par NaN pour l'imputation
    X_input = X_input.replace(-1, pd.NA)

    X_prep = preprocessor.transform(X_input)
    return X_prep


def predict_implantation_station(csv_path: str) -> pd.DataFrame | None:
    """Charge un CSV, le prétraite et prédit la colonne 'implantation_station'.

    Args:
        csv_path: Chemin vers le fichier CSV d'entrée.

    Returns:
        DataFrame original avec la colonne 'predicted_implantation_station' ajoutée,
        ou None en cas d'erreur.
    """
    preprocessor, ordinal_encoder, model = load_model_and_preprocessor()
    if preprocessor is None or ordinal_encoder is None or model is None:
        return None

    try:
        df_new = pd.read_csv(csv_path, encoding="UTF-8")
        print(f"CSV chargé avec {len(df_new)} lignes.")
    except FileNotFoundError:
        print(f"Erreur: Le fichier CSV n'a pas été trouvé à l'adresse {csv_path}")
        return None

    df_for_preprocessing = df_new.copy()

    # Encodage des colonnes catégorielles avec l'OrdinalEncoder d'entraînement
    cat_cols_present = [c for c in CAT_COLS if c in df_for_preprocessing.columns]
    if cat_cols_present:
        df_for_preprocessing[cat_cols_present] = ordinal_encoder.transform(
            df_for_preprocessing[cat_cols_present]
        )

    X_new_prep = preprocess_new_data(df_for_preprocessing, preprocessor)

    predictions = model.predict(X_new_prep)
    df_new["predicted_implantation_station"] = predictions
    return df_new



#csv_path = "./Besion_Client_3/csv.csv"
#
#df_with_predictions = predict_implantation_station(csv_path)
#
# if df_with_predictions is not None:
#     print(df_with_predictions[['date_mise_en_service', 'horaires', 'predicted_implantation_station']].head())