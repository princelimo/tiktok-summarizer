"""
Application FastAPI pour résumer des vidéos TikTok.
Extrait les sous-titres via yt-dlp et génère un résumé structuré via l'API Gemini.
"""

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import os

_gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
print("Clé Gemini chargée :", "Oui" if _gemini_key else "Non", flush=True)

import json
import tempfile
from typing import Any

import yt_dlp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import google.generativeai as genai

app = FastAPI(
    title="TikTok Summarizer",
    description="Extrait les sous-titres d'une vidéo TikTok et génère un résumé structuré.",
    version="1.1.0",
)


@app.on_event("startup")
def _log_gemini_status():
    statut = "Oui" if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) else "Non"
    print(f"Clé Gemini chargée : {statut}", flush=True)


class SummarizeRequest(BaseModel):
    url: HttpUrl


class SummaryResponse(BaseModel):
    titre: str
    auteur: str
    idee_principale: str
    points_cles: list[str]
    conclusion: str
    tags: list[str]


def extraire_info(url: str) -> dict | None:
    """
    Utilise yt-dlp pour extraire les sous-titres ET les métadonnées (auteur).
    Retourne un dict { "transcript": str, "auteur": str } ou None.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["fr", "en", "fr.*", "en.*"],
            "subtitlesformat": "vtt/srt/best",
            "outtmpl": os.path.join(tmpdir, "subs"),
            "quiet": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if not info:
                    return None

                # Récupération de l'auteur depuis les métadonnées yt-dlp
                auteur = (
                    info.get("uploader")
                    or info.get("uploader_id")
                    or info.get("channel")
                    or "Auteur inconnu"
                )
                # Ajouter @ si pas déjà présent
                if not auteur.startswith("@"):
                    auteur = f"@{auteur}"

                # Récupération du transcript
                transcript = None
                for f in os.listdir(tmpdir):
                    if f.endswith((".vtt", ".srt", ".ass")):
                        path = os.path.join(tmpdir, f)
                        with open(path, encoding="utf-8", errors="replace") as fp:
                            transcript = fp.read()
                        break

                return {"transcript": transcript, "auteur": auteur}
        except Exception:
            return None


def nettoyer_transcript(texte: str) -> str:
    """Supprime les balises VTT/SRT et numéros pour ne garder que le texte."""
    if not texte:
        return ""
    lignes = []
    for line in texte.splitlines():
        line = line.strip()
        if not line or line.isdigit():
            continue
        if "-->" in line or line.startswith("WEBVTT") or line.startswith("Kind:"):
            continue
        lignes.append(line)
    return " ".join(lignes).strip()


def resumer_avec_gemini(transcript: str, auteur: str) -> SummaryResponse:
    """
    Envoie le transcript à Gemini et retourne un résumé structuré avec tags précis.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY non configurée")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-3-flash-preview")

    prompt = f"""Tu es un assistant qui résume des transcriptions de vidéos TikTok à contenu politique et social.
Résume la transcription suivante en français de manière structurée.
Réponds UNIQUEMENT avec un objet JSON valide, sans markdown ni texte autour, avec exactement ces clés :
- "titre" : un titre court et précis pour la vidéo
- "idee_principale" : l'idée principale en une ou deux phrases
- "points_cles" : une liste de 3 à 5 points clés (chaque élément une phrase complète)
- "conclusion" : une phrase de conclusion
- "tags" : une liste de 3 à 6 tags TRÈS PRÉCIS sur les thèmes abordés. 
  Utilise des tags spécifiques comme : féminisme, géopolitique, antiracisme, islamophobie, 
  droits LGBTQ+, immigration, capitalisme, décolonisation, médias, violences policières, 
  classe sociale, écologie politique, sexisme, palestine, racisme systémique, laïcité, 
  nationalisme, précarité, santé mentale, validisme, grossophobie, transphobie, 
  antisémitisme, libertés civiles, surveillance, désinformation, néolibéralisme, etc.
  Choisis uniquement les tags qui correspondent vraiment au contenu.

Transcription :
{transcript}
"""

    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise ValueError("Réponse Gemini vide")

        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data: dict[str, Any] = json.loads(text)
        return SummaryResponse(
            titre=data.get("titre", ""),
            auteur=auteur,
            idee_principale=data.get("idee_principale", ""),
            points_cles=data.get("points_cles", []),
            conclusion=data.get("conclusion", ""),
            tags=data.get("tags", []),
        )
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=502, detail=f"Réponse Gemini invalide (JSON) : {e!s}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erreur API Gemini : {e!s}")


@app.post("/summarize", response_model=SummaryResponse)
def summarize(request: SummarizeRequest) -> SummaryResponse:
    url = str(request.url)

    # Étape 1 : extraction transcript + auteur
    info = extraire_info(url)
    if not info or not info.get("transcript"):
        raise HTTPException(status_code=404, detail="Sous-titres non disponibles pour cette vidéo")

    transcript = nettoyer_transcript(info["transcript"])
    if not transcript:
        raise HTTPException(status_code=404, detail="Sous-titres non disponibles pour cette vidéo")

    auteur = info.get("auteur", "Auteur inconnu")

    # Étape 2 : résumé + tags via Gemini
    return resumer_avec_gemini(transcript, auteur)


@app.exception_handler(HTTPException)
def http_exception_handler(_request, exc: HTTPException):
    from fastapi.responses import JSONResponse
    if exc.detail == "Sous-titres non disponibles pour cette vidéo":
        return JSONResponse(status_code=404, content={"error": "Sous-titres non disponibles pour cette vidéo"})
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail} if isinstance(exc.detail, str) else exc.detail,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
