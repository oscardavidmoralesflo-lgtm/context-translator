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

# Sesión HTTP persistente con Keep-Alive y Connection Pooling para máxima velocidad
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
session.mount("https://", adapter)

# Modelo directo de alta velocidad
ACTIVE_MODEL = "models/gemini-3.6-flash"

SYSTEM_PROMPT = """Eres un lingüista y traductor contextual de élite. Sé conciso, directo y rápido.
Responde ÚNICAMENTE en JSON con esta estructura exacta:
{
  "main_translation": "Traducción más natural",
  "ipa_target": "IPA",
  "all_meanings": [
    {
      "context_category": "Coloquial / Diario",
      "translation": "Traducción coloquial",
      "example_usage": "Ejemplo corto",
      "nuance": "Uso clave"
    },
    {
      "context_category": "Formal / Profesional",
      "translation": "Traducción formal",
      "example_usage": "Ejemplo formal",
      "nuance": "Uso clave"
    },
    {
      "context_category": "Slang / Modismo / Otros",
      "translation": "Significado o modismo",
      "example_usage": "Ejemplo corto",
      "nuance": "Uso clave"
    }
  ],
  "pronunciation_tip": "Tip fonético conciso (1 frase)"
}"""

@app.get("/")
async def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    elif os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return HTMLResponse("<h2>Context Translator Live</h2>")

def fetch_gemini_response(api_key: str, user_query: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/{ACTIVE_MODEL}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
            "maxOutputTokens": 700
        }
    }
    return session.post(url, headers=headers, json=payload, timeout=15)

@app.post("/api/translate")
async def translate(req: TranslationRequest):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío.")

    user_query = f"{SYSTEM_PROMPT}\n\nTexto a analizar: \"{req.text}\"\nOrigen: {req.source_lang}\nDestino: {req.target_lang}\nTono: {req.tone_preference}"

    try:
        # Llamada HTTP en hilo no bloqueante
        res = await asyncio.to_thread(fetch_gemini_response, api_key, user_query)

        if res.status_code != 200:
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
        raise HTTPException(status_code=500, detail="Respuesta del modelo inválida.")
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
