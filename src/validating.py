import os
import shutil
import mlflow
from mlflow.tracking import MlflowClient

# ============================================================
# 1. Configuration
# ============================================================

EXPERIMENT_NAME = "training_model_Esperance_de_vie"
REGISTERED_MODEL_NAME = "LifeExpectancy_Champion"

# Choix de la métrique pour sélectionner le modèle
METRIC_NAME = "val_r2"        # Vous pouvez aussi utiliser "val_rmse" ou "val_mae"
GREATER_IS_BETTER = True      # True pour R2 (plus haut est meilleur), False pour RMSE/MAE

# Liste des modèles entraînés
MODEL_NAMES = ["ElasticNet", "RandomForest", "GradientBoosting"]

os.makedirs("models", exist_ok=True)


# ============================================================
# 2. Recherche du Champion dans MLflow
# ============================================================

client = MlflowClient()

# Récupération de l'expérience MLflow
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    raise ValueError(f"Expérience '{EXPERIMENT_NAME}' introuvable dans MLflow.")

print("=" * 60)
print(f"SELECTION DU CHAMPION (Métrique clé : {METRIC_NAME})")
print("=" * 60)

best_run = None
best_score = float("-inf") if GREATER_IS_BETTER else float("inf")
best_model_name = None

# Parcourir le dernier run MLflow de chaque modèle
for model_name in MODEL_NAMES:
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model_type = '{model_name}'",
        order_by=["attribute.start_time DESC"],
        max_results=1
    )

    if not runs:
        print(f"⚠️ Aucun run trouvé pour {model_name}")
        continue

    run = runs[0]
    score = run.data.metrics.get(METRIC_NAME)

    if score is None:
        print(f"⚠️ Métrique '{METRIC_NAME}' non trouvée pour {model_name}")
        continue

    print(f"Modèle: {model_name:18} | Run ID: {run.info.run_id} | {METRIC_NAME}: {score:.4f}")

    # Comparaison du score
    is_better = (score > best_score) if GREATER_IS_BETTER else (score < best_score)
    if is_better:
        best_score = score
        best_run = run
        best_model_name = model_name

if not best_run:
    raise RuntimeError("Aucun modèle valide n'a été trouvé.")

print("\n" + "=" * 60)
print(f"🏆 CHAMPION SÉLECTIONNÉ : {best_model_name}")
print(f"📌 Run ID MLflow        : {best_run.info.run_id}")
print(f"📊 Score {METRIC_NAME:12} : {best_score:.4f}")
print("=" * 60)


# ============================================================
# 3. Promotion dans MLflow Model Registry (Alias @Champion)
# ============================================================

model_uri = f"runs:/{best_run.info.run_id}/model"
model_version = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="Champion",
    version=model_version.version
)
print(f"✅ Alias '@Champion' attribué à la version {model_version.version} dans MLflow Registry")


# ============================================================
# 4. Export de l'artefact local pour DVC / CI-CD
# ============================================================

# Détermination du nom du fichier joblib produit lors de l'entraînement
selected_joblib_filename = best_model_name.lower().replace(" ", "_") + ".joblib"
selected_joblib_path = os.path.join("models", selected_joblib_filename)

champion_path = os.path.join("models", "champion_model.joblib")

# Copie directe du modèle sélectionné vers 'champion_model.joblib'
shutil.copyfile(selected_joblib_path, champion_path)
print(f"📦 Artefact du modèle Champion copié avec succès dans : {champion_path}")