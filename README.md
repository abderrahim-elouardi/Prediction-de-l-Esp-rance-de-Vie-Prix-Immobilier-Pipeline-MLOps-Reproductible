#  Prediction de l'Espérance de Vie & Prix Immobilier – Pipeline MLOps Reproductible

<!-- [![CI Workflow](https://github.com/votre-user/mlops-house-price/actions/workflows/ci.yml/badge.svg)](https://github.com/votre-user/mlops-house-price/actions)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![DVC Tracked](https://img.shields.io/badge/DVC-tracked-white.svg?logo=dvc)](https://dvc.org/)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-tracking-blue?logo=mlflow)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://www.docker.com/) -->

Ce projet implémente un système Machine Learning industriel et reproductible de bout en bout pour la prédiction de prix/données immobilières. L'objectif principal est de mettre en œuvre les meilleures pratiques **MLOps** : versionnement des données et pipelines avec **DVC**, suivi des expériences avec **MLflow**, conteneurisation avec **Docker**, et intégration continue (CI) automatisée via **GitHub Actions**.

---

##  Architecture de la Solution

Le système sépare rigoureusement la CI rapide, le pipeline d'entraînement lourd et la livraison applicative :

```text
               ┌──────────────────────────────────────────────┐
               │              Git & GitHub                    │
               └──────────────────────┬───────────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
            ┌────────────────────────┐  ┌────────────────────────┐
            │   GitHub Actions CI    │  ┌  DVC Data Pipeline  │
            ├────────────────────────┤  ├────────────────────────┤
            │ • Ruff (Lint)          │  │ • Data Validation      │
            │ • Pytest & Coverage    │  │ • Preprocessing        │
            │ • Docker Build Test    │  │ • Model Training       │
            └────────────┬───────────┘  └────────────┬───────────┘
                         │                           │
                         ▼                           ▼
                 Code Validé                 MLflow Tracking
                                            (Params / Metrics / Model)


## Structure du Projet
mlops-house-price/
├── .github/
│   └── workflows/
│       ├── ci.yml               # Pipeline CI rapide (Lint + Tests + Docker Build)
│       └── train.yml            # Pipeline d'entraînement automatisé
├── data/
│   ├── raw/                      # Données brutes (versionnées par DVC)
│   └── processed/                # Données nettoyées et préparées
├── models/                       # Artefacts des modèles entraînés (.joblib)
├── src/                          # Code source modulaire
│   ├── validate_data.py          # Contrôle qualité et schémas de données
│   ├── preprocess.py             # Nettoyage et OneHot/Scaling
│   ├── train.py                  # Entraînement & logging MLflow
│   ├── evaluate.py               # Calcul des métriques d'évaluation
│   └── main.py                   # API REST FastAPI pour le modèle
├── tests/                        # Suite de tests unitaires et d'intégration[cite: 5]
│   ├── test_validation.py[cite: 5]
│   ├── test_preprocess.py[cite: 5]
│   └── test_api.py[cite: 5]
├── dvc.yaml                      # Définition des stages du pipeline DVC[cite: 5]
├── params.yaml                   # Hyperparamètres et configurations[cite: 5]
├── pyproject.toml                # Configurations ruff et pytest[cite: 5]
├── requirements.txt              # Dépendances du projet[cite: 5]
├── Dockerfile                    # Containerisation de l'API de prédiction[cite: 5]
└── README.md