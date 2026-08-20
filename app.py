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

SYSTEM_PROMPT = """
Eres un lingüista y traductor contextual de élite. Analiza la palabra o frase y desglosa todos sus significados y contextos.
Responde ÚNICAMENTE con un JSON válido con esta estructura exacta (sin texto previo ni bloques markdown):
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
    return HTMLResponse("<h2>Error: No se encontró index.html</h2>")

def get_active_model_name(api_key: str) -> str:
    """Consulta los modelos activos de tu cuenta en tiempo real."""
    headers = {"x-goog-api-key": api_key}
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            models = data.get("models", [])
            supported = [
                m["name"] for m in models 
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
            print(f"[DEBUG] Modelos activos en tu clave: {supported}", flush=True)

            # Prioridad 1: Versión 3.6
            for m in supported:
                if "3.6" in m:
                    return m
            # Prioridad 2: Cualquier versión Flash activa
            for m in supported:
                if "flash" in m.lower():
                    return m
            # Prioridad 3: El primer modelo disponible
            if supported:
                return supported[0]
    except Exception as e:
        print(f"[DEBUG] Error listando modelos: {e}", flush=True)

    # Fallback directo al modelo oficial vigente
    return "models/gemini-3.6-flash"

@app.post("/api/translate")
async def translate(req: TranslationRequest):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada en Render.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío.")

    model_identifier = get_active_model_name(api_key)
    if not model_identifier.startswith("models/"):
        model_identifier = f"models/{model_identifier}"

    url = f"https://generativelanguage.googleapis.com/v1beta/{model_identifier}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    user_query = f"{SYSTEM_PROMPT}\n\nTexto a analizar: \"{req.text}\"\nOrigen: {req.source_lang}\nDestino: {req.target_lang}\nTono: {req.tone_preference}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=25)
        if res.status_code != 200:
            print(f"[ERROR] Llamada a {url} falló: {res.status_code} - {res.text}", flush=True)
            raise HTTPException(status_code=500, detail=f"Error Gemini API ({res.status_code}): {res.text}")

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
        raise HTTPException(status_code=500, detail="El modelo no devolvió un JSON estructurado.")
    except Exception as e:
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
