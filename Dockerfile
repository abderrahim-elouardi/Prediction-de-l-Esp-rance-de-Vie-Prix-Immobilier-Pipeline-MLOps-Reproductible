# 1. Utiliser l'image Debian Slim officielle pour uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER root

# Désactiver la vérification des dates de validité Debian pour éviter les bugs de Time Drift
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && \
    apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copier la configuration des dépendances uv
COPY pyproject.toml uv.lock* ./

# Installer les dépendances Python (wheels précompilés scikit-learn/scipy)
RUN uv sync --frozen --no-cache || uv sync --no-cache

# Ajouter le venv au PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copier le projet
COPY . .

# Commande par défaut pour lancer le pipeline DVC
CMD ["uv", "run", "dvc", "repro", "--no-commit"]