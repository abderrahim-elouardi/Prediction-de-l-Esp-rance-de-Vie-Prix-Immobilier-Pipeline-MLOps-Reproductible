import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# 1. Configuration & Chemins
# ============================================================
TEST_X_PATH = "data/processed/test_df_x.csv"
TEST_Y_PATH = "data/processed/test_df_y.csv"
MODEL_PATH = "models/champion_model.joblib"

METRICS_DIR = "metrics"
PLOT_OUTPUT_PATH = "metrics/residuals_plot.png"
JSON_OUTPUT_PATH = "metrics/test_metrics.json"


EXPERIMENT_NAME = "training_model_Esperance_de_vie"
REGISTERED_MODEL_NAME = "LifeExpectancy_Champion"

os.makedirs(METRICS_DIR, exist_ok=True)


# ============================================================
# 2. Fonction de génération du Residuals Plot
# ============================================================

def plot_residuals(y_true, y_pred, output_path):
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, color='blue', edgecolors='k')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel("Valeurs prédites (y_pred)")
    plt.ylabel("Résidus (y_real - y_pred)")
    plt.title("Residuals Plot - Champion Model")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Residuals Plot sauvegardé dans : {output_path}")


# ============================================================
# 3. Évaluation sur le Jeu de Test
# ============================================================

print("=" * 60)
print("EVALUATION DU MODELE CHAMPION SUR LE JEU DE TEST")
print("=" * 60)

# Chargement des données et du modèle
df_test_x = pd.read_csv(TEST_X_PATH)
df_test_y = pd.read_csv(TEST_Y_PATH).values.ravel()  # Conversion en 1D

champion_model = joblib.load(MODEL_PATH)

# Prédictions
y_pred = champion_model.predict(df_test_x)

# Calcul des métriques de test
test_rmse = np.sqrt(mean_squared_error(df_test_y, y_pred))
test_mae = mean_absolute_error(df_test_y, y_pred)
test_r2 = r2_score(df_test_y, y_pred)

print(f"Test RMSE : {test_rmse:.4f}")
print(f"Test MAE  : {test_mae:.4f}")
print(f"Test R²   : {test_r2:.4f}")

# Génération du graphique des résidus
plot_residuals(df_test_y, y_pred, PLOT_OUTPUT_PATH)

# Sauvegarde des métriques locales pour DVC
test_metrics = {
    "test_rmse": float(test_rmse),
    "test_mae": float(test_mae),
    "test_r2": float(test_r2)
}

with open(JSON_OUTPUT_PATH, "w") as f:
    json.dump(test_metrics, f, indent=4)
print(f"📄 Métriques de test sauvegardées dans : {JSON_OUTPUT_PATH}")

# Loguer les résultats dans la version Champion sur MLflow
try:
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "Champion")
    
    with mlflow.start_run(run_id=model_version.run_id):
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.log_artifact(PLOT_OUTPUT_PATH)
        
    print("✅ Métriques et graphique enregistrés sous le Run Champion dans MLflow !")
except Exception as e:
    print(f"⚠️ Warning MLflow : {e}")

