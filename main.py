"""
Application FastAPI pour résumer des vidéos TikTok.
Extrait les sous-titres via yt-dlp et génère un résumé structuré via l'API Gemini.
"""

from pathlib import Path

from dotenv import load_dotenv

# Charger le .env depuis le dossier du projet (fiable même si le cwd change)
load_dotenv(Path(__file__).resolve().parent / ".env")

import os

# Vérification visible au démarrage (sans afficher la clé)
_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
print("Clé Gemini chargée :", "Oui" if _gemini_key else "Non", flush=True)

import json
import tempfile
from typing import Any

import yt_dlp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import google.generativeai as genai

# Configuration de l'application FastAPI
app = FastAPI(
    title="TikTok Summarizer",
    description="Extrait les sous-titres d'une vidéo TikTok et génère un résumé structuré.",
    version="1.0.0",
)


@app.on_event("startup")
def _log_gemini_status():
    """Affiche le statut de la clé Gemini dans les logs uvicorn au démarrage."""
    statut = "Oui" if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) else "Non"
    print(f"Clé Gemini chargée : {statut}", flush=True)


# Modèle pour la requête entrante
class SummarizeRequest(BaseModel):
    """Corps de la requête : URL TikTok à résumer."""

    url: HttpUrl


# Modèle pour le résumé structuré retourné par Gemini
class SummaryResponse(BaseModel):
    """Résumé structuré : titre, idée principale, points clés, conclusion."""

    titre: str
    idee_principale: str
    points_cles: list[str]
    conclusion: str


def extraire_transcript(url: str) -> str | None:
    """
    Utilise yt-dlp pour extraire les sous-titres/transcript de la vidéo.
    Retourne le texte du transcript ou None si non disponible.
    """
    # Répertoire temporaire pour les sous-titres
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "skip_download": True,  # On ne télécharge pas la vidéo
            "writesubtitles": True,
            "writeautomaticsub": True,  # Sous-titres auto si pas de sous-titres manuels
            "subtitleslangs": ["fr", "en", "fr.*", "en.*"],  # Priorité français puis anglais
            "subtitlesformat": "vtt/srt/best",
            "outtmpl": os.path.join(tmpdir, "subs"),
            "quiet": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    return None
                # Chercher le fichier de sous-titres généré
                for f in os.listdir(tmpdir):
                    if f.endswith((".vtt", ".srt", ".ass")):
                        path = os.path.join(tmpdir, f)
                        with open(path, encoding="utf-8", errors="replace") as fp:
                            return fp.read()
        except Exception:
            return None
    return None


def nettoyer_transcript(texte: str) -> str:
    """Supprime les balises VTT/SRT et numéros pour ne garder que le texte."""
    if not texte:
        return ""
    lignes = []
    for line in texte.splitlines():
        line = line.strip()
        # Ignorer les timestamps et numéros de séquence
        if not line or line.isdigit():
            continue
        if "-->" in line or line.startswith("WEBVTT") or line.startswith("Kind:"):
            continue
        lignes.append(line)
    return " ".join(lignes).strip()


def resumer_avec_gemini(transcript: str) -> SummaryResponse:
    """
    Envoie le transcript à l'API Gemini et retourne un résumé structuré en français.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY ou GOOGLE_API_KEY non configurée",
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")

    prompt = f"""Tu es un assistant qui résume des transcriptions de vidéos TikTok.
Résume la transcription suivante en français de manière structurée.
Réponds UNIQUEMENT avec un objet JSON valide, sans markdown ni texte autour, avec exactement ces clés :
- "titre" : un titre court pour la vidéo
- "idee_principale" : l'idée principale en une ou deux phrases
- "points_cles" : une liste de points clés (chaque élément une phrase)
- "conclusion" : une phrase de conclusion

Transcription :
{transcript}
"""

    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise ValueError("Réponse Gemini vide")
        # Nettoyer d'éventuels blocs markdown (```json ... ```)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data: dict[str, Any] = json.loads(text)
        return SummaryResponse(
            titre=data.get("titre", ""),
            idee_principale=data.get("idee_principale", ""),
            points_cles=data.get("points_cles", []),
            conclusion=data.get("conclusion", ""),
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Réponse Gemini invalide (JSON) : {e!s}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur API Gemini : {e!s}",
        )


@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummarizeRequest) -> SummaryResponse:
    """
    Endpoint principal : reçoit une URL TikTok, extrait les sous-titres,
    les envoie à Gemini et retourne un résumé structuré en JSON.
    """
    url = str(request.url)

    # Étape 1 : extraction du transcript via yt-dlp
    transcript_brut = extraire_transcript(url)
    if not transcript_brut:
        raise HTTPException(
            status_code=404,
            detail="Sous-titres non disponibles pour cette vidéo",
        )

    transcript = nettoyer_transcript(transcript_brut)
    if not transcript:
        raise HTTPException(
            status_code=404,
            detail="Sous-titres non disponibles pour cette vidéo",
        )

    # Étape 2 : résumé via Gemini
    return resumer_avec_gemini(transcript)


# Gestion explicite du message d'erreur attendu en JSON
@app.exception_handler(HTTPException)
def http_exception_handler(_request, exc: HTTPException):
    """Retourne les erreurs HTTP en JSON (dont 'Sous-titres non disponibles')."""
    from fastapi.responses import JSONResponse

    if exc.detail == "Sous-titres non disponibles pour cette vidéo":
        return JSONResponse(
            status_code=404,
            content={"error": "Sous-titres non disponibles pour cette vidéo"},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail} if isinstance(exc.detail, str) else exc.detail,
    )


@app.get("/health")
def health():
    """Endpoint de santé pour Cloud Run (readiness/liveness)."""
    return {"status": "ok"}
