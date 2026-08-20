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

# Sesión HTTP persistente para máxima velocidad
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
session.mount("https://", adapter)

# Modelos oficiales de la serie Gemini 3
MODELS_TO_TRY = [
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

WORKING_MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """Eres un lingüista y traductor contextual de élite. Analiza la palabra o frase y desglosa sus significados y contextos de forma concisa y directa.
Responde ÚNICAMENTE con un JSON válido con esta estructura exacta (sin texto adicional ni explicaciones fuera del JSON):
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
  "pronunciation_tip": "Consejo fonético o de entonación conciso"
}"""

@app.get("/")
async def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    elif os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return HTMLResponse("<h2>Context Translator Live</h2>")

def clean_json_response(raw_text: str) -> dict:
    text = raw_text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)

def call_gemini_api(api_key: str, prompt: str):
    global WORKING_MODEL

    # Intentar primero con el modelo que funcionó, luego con las variantes Gemini 3
    candidates = [WORKING_MODEL] + [m for m in MODELS_TO_TRY if m != WORKING_MODEL]

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
            "maxOutputTokens": 800
        }
    }

    last_error = ""
    for model in candidates:
        clean_model = model.replace("models/", "")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent"
        
        try:
            res = session.post(url, headers=headers, json=payload, timeout=15)
            if res.status_code == 200:
                WORKING_MODEL = clean_model
                data = res.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                return clean_json_response(raw_text)
            else:
                last_error = f"{clean_model} (HTTP {res.status_code}): {res.text}"
                print(f"[WARN] Falló {clean_model}: {res.status_code} - {res.text}", flush=True)
        except Exception as e:
            last_error = f"{clean_model} error: {str(e)}"
            print(f"[WARN] Error con {clean_model}: {e}", flush=True)

    raise Exception(f"No se pudo completar con Gemini. Detalle: {last_error}")

@app.post("/api/translate")
async def translate(req: TranslationRequest):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada en Render.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío.")

    user_query = f"{SYSTEM_PROMPT}\n\nTexto a analizar: \"{req.text}\"\nOrigen: {req.source_lang}\nDestino: {req.target_lang}\nTono: {req.tone_preference}"

    try:
        result = await asyncio.to_thread(call_gemini_api, api_key, user_query)
        return result
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
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
