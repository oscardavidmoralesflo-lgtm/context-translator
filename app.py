import os
import json
import asyncio
import requests
import edge_tts
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Contextual Neural Translator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"
    tone_preference: str = "natural"

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-ChristopherNeural"

# Sesión HTTP persistente
session = requests.Session()

# Memoria caché para no consultar los modelos en cada clic
CACHED_MODEL_CONFIG = {
    "version": "v1beta",
    "model_name": None
}

SYSTEM_PROMPT = """
Eres un lingüista y traductor contextual de élite. Analiza la palabra o frase y desglosa todos sus significados y contextos de forma concisa y directa.
Responde ÚNICAMENTE con un JSON válido con esta estructura exacta:
{
  "main_translation": "Traducción principal más precisa",
  "ipa_target": "Transcripción fonética IPA de la traducción principal",
  "all_meanings": [
    {
      "context_category": "Coloquial / Diario",
      "translation": "Traducción en este contexto",
      "example_usage": "Ejemplo corto de uso",
      "nuance": "Explicación de cuándo usarlo"
    },
    {
      "context_category": "Formal / Profesional",
      "translation": "Traducción formal",
      "example_usage": "Ejemplo en contexto laboral/académico",
      "nuance": "Uso adecuado"
    },
    {
      "context_category": "Slang / Modismo / Otros",
      "translation": "Significado alternativo o modismo",
      "example_usage": "Ejemplo de uso",
      "nuance": "Matiz cultural o regional"
    }
  ],
  "pronunciation_tip": "Consejo fonético o de entonación"
}
"""

@app.get("/")
async def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    elif os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return HTMLResponse("<h2>Context Translator Live</h2>")

def get_or_detect_model(api_key: str):
    """Detecta el modelo compatible en tu cuenta una sola vez y lo guarda en memoria."""
    global CACHED_MODEL_CONFIG
    if CACHED_MODEL_CONFIG["model_name"]:
        return CACHED_MODEL_CONFIG["version"], CACHED_MODEL_CONFIG["model_name"]

    headers = {"x-goog-api-key": api_key}
    for version in ["v1beta", "v1"]:
        try:
            url = f"https://generativelanguage.googleapis.com/{version}/models"
            res = session.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                models = data.get("models", [])
                supported = [
                    m["name"] for m in models 
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                print(f"[INFO] Modelos activos en tu cuenta: {supported}", flush=True)

                selected = None
                # Prioridad 1: Modelos Flash
                for m in supported:
                    if "flash" in m.lower():
                        selected = m
                        break
                # Prioridad 2: Cualquier modelo activo
                if not selected and supported:
                    selected = supported[0]

                if selected:
                    CACHED_MODEL_CONFIG["version"] = version
                    CACHED_MODEL_CONFIG["model_name"] = selected
                    print(f"[INFO] Modelo fijado en memoria: {selected} ({version})", flush=True)
                    return version, selected
        except Exception as e:
            print(f"[WARN] Error detectando en {version}: {e}", flush=True)

    # Fallback si no pudo autodescubrir
    fallback = "models/gemini-1.5-flash"
    CACHED_MODEL_CONFIG["version"] = "v1beta"
    CACHED_MODEL_CONFIG["model_name"] = fallback
    return "v1beta", fallback

@app.post("/api/translate")
async def translate(req: TranslationRequest):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada en Render.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío.")

    # Obtener modelo (instantáneo desde la caché de memoria)
    api_version, model_name = get_or_detect_model(api_key)

    url = f"https://generativelanguage.googleapis.com/{api_version}/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    user_query = f"{SYSTEM_PROMPT}\n\nTexto a analizar: \"{req.text}\"\nOrigen: {req.source_lang}\nDestino: {req.target_lang}\nTono: {req.tone_preference}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2
        }
    }

    try:
        # Llamada HTTP no bloqueante y directa
        res = await asyncio.to_thread(session.post, url, headers=headers, json=payload, timeout=20)

        if res.status_code != 200:
            print(f"[ERROR] Gemini respondió {res.status_code}: {res.text}", flush=True)
            # Resetear memoria por si el modelo cambió
            CACHED_MODEL_CONFIG["model_name"] = None
            raise HTTPException(status_code=500, detail=f"Error Gemini ({res.status_code}): {res.text}")

        data = res.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        return json.loads(raw_text.strip())

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="El modelo no devolvió un JSON válido.")
    except Exception as e:
        print(f"[EXCEPTION] {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tts")
async def generate_speech(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Texto vacío.")
    try:
        communicate = edge_tts.Communicate(req.text, req.voice)
        audio_stream = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.extend(chunk["data"])
        return Response(content=bytes(audio_stream), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
