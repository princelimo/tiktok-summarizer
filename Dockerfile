# Image Python légère pour Cloud Run
FROM python:3.12-slim-bookworm

# Variable d'environnement pour éviter les fichiers .pyc et buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Répertoire de l'application
WORKDIR /app

# Installation des dépendances système éventuelles (yt-dlp peut en avoir besoin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copie des dépendances Python d'abord (meilleure utilisation du cache Docker)
COPY requirements.txt .

# Installation des dépendances dans un venv ou directement
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY main.py .

# Cloud Run expose le port 8080 par défaut
EXPOSE 8080

# Démarrer avec uvicorn (Cloud Run injecte PORT, défaut 8080)
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]
