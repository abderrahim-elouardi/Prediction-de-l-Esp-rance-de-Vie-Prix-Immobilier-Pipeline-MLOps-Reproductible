from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os


# ============================================================
# 1. Chargement des données
# ============================================================

df_train_x = pd.read_csv("data/processed/train_df_x.csv")
df_train_y = pd.read_csv("data/processed/train_df_y.csv")


# ============================================================
# 2. Création d'un jeu de validation
# ============================================================

df_train_x, df_val_x, df_train_y, df_val_y = train_test_split(
    df_train_x,
    df_train_y,
    random_state=123,
    test_size=0.15
)


# ============================================================
# 3. Définition des modèles
# ============================================================

models = {

    "ElasticNet": ElasticNet(
        alpha=1.0,
        l1_ratio=0.5,
        fit_intercept=True,
        max_iter=1000,
        random_state=123
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=123,
        n_jobs=-1
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=123
    )
}


# ============================================================
# 4. Création du dossier models
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# 5. Expérience MLflow
# ============================================================

mlflow.set_experiment(
    experiment_name="training_model_Esperance_de_vie"
)


# ============================================================
# 6. Entraînement des modèles
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"Training : {model_name}")
    print("=" * 60)

    with mlflow.start_run(run_name=model_name):

        # ----------------------------------------------------
        # Entraînement
        # ----------------------------------------------------

        model.fit(df_train_x, df_train_y.values.ravel())


        # ----------------------------------------------------
        # Prédictions
        # ----------------------------------------------------

        train_predicted_y = model.predict(df_train_x)
        val_predicted_y = model.predict(df_val_x)


        # ----------------------------------------------------
        # Métriques TRAIN
        # ----------------------------------------------------

        train_rmse = np.sqrt(
            mean_squared_error(
                df_train_y,
                train_predicted_y
            )
        )

        train_mae = mean_absolute_error(
            df_train_y,
            train_predicted_y
        )

        train_r2 = r2_score(
            df_train_y,
            train_predicted_y
        )


        # ----------------------------------------------------
        # Métriques VALIDATION
        # ----------------------------------------------------

        val_rmse = np.sqrt(
            mean_squared_error(
                df_val_y,
                val_predicted_y
            )
        )

        val_mae = mean_absolute_error(
            df_val_y,
            val_predicted_y
        )

        val_r2 = r2_score(
            df_val_y,
            val_predicted_y
        )


        # ----------------------------------------------------
        # Affichage
        # ----------------------------------------------------

        print(f"Train RMSE : {train_rmse:.4f}")
        print(f"Train MAE  : {train_mae:.4f}")
        print(f"Train R²   : {train_r2:.4f}")

        print(f"Val RMSE   : {val_rmse:.4f}")
        print(f"Val MAE    : {val_mae:.4f}")
        print(f"Val R²     : {val_r2:.4f}")


        # ----------------------------------------------------
        # MLflow
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            model_name
        )

        # Log des paramètres du modèle
        mlflow.log_params(
            model.get_params()
        )

        # Métriques
        mlflow.log_metric(
            "train_rmse",
            train_rmse
        )

        mlflow.log_metric(
            "train_mae",
            train_mae
        )

        mlflow.log_metric(
            "train_r2",
            train_r2
        )

        mlflow.log_metric(
            "val_rmse",
            val_rmse
        )

        mlflow.log_metric(
            "val_mae",
            val_mae
        )

        mlflow.log_metric(
            "val_r2",
            val_r2
        )


        # ----------------------------------------------------
        # Sauvegarde MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )


        # ----------------------------------------------------
        # Sauvegarde locale
        # ----------------------------------------------------

        model_filename = (
            model_name.lower().replace(" ", "_")
            + ".joblib"
        )

        model_path = os.path.join(
            "models",
            model_filename
        )

        joblib.dump(
            model,
            model_path
        )

        print(f"Model saved : {model_path}")
    