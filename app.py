import os
import json
import asyncio
import requests
import edge_tts
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Contextual Reverso Translator")

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
    target_lang: str = "es"
    tone_preference: str = "natural"

class TTSRequest(BaseModel):
    text: str
    voice: str = "en-US-ChristopherNeural"

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=20)
session.mount("https://", adapter)

MODELS_TO_TRY = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash"]
WORKING_MODEL = "gemini-3.6-flash"

SYSTEM_PROMPT = """Eres un motor de traducción contextual avanzado al estilo 'Reverso Context'.
Analiza el término o frase y entrega un corpus bilingüe con ejemplos reales.

Responde ÚNICAMENTE con un JSON válido con esta estructura:
{
  "query": "término buscado",
  "main_translation": "Traducción principal más natural",
  "part_of_speech": "Tipo de palabra (Verb, Phrasal Verb, Noun, Idiom, Adj)",
  "ipa_us": "Transcripción IPA en inglés americano",
  "ipa_uk": "Transcripción IPA en inglés británico",
  "category_topic": "Temática general (ej: Relationships & Social Interactions)",
  "translations_chips": [
    "traducción 1", "traducción 2", "traducción 3", "traducción 4", "traducción 5", "traducción 6"
  ],
  "context_examples": [
    {
      "source_sentence": "Oración en idioma origen",
      "source_highlight": "fragmento a resaltar en origen",
      "target_sentence": "Oración traducida en destino",
      "target_highlight": "fragmento a resaltar en destino",
      "context_label": "Contexto (ej: Coloquial, Convivencia, Negocios)"
    },
    {
      "source_sentence": "Oración 2 en origen",
      "source_highlight": "fragmento",
      "target_sentence": "Oración 2 en destino",
      "target_highlight": "fragmento",
      "context_label": "Contexto 2"
    },
    {
      "source_sentence": "Oración 3 en origen",
      "source_highlight": "fragmento",
      "target_sentence": "Oración 3 en destino",
      "target_highlight": "fragmento",
      "context_label": "Contexto 3"
    },
    {
      "source_sentence": "Oración 4 en origen",
      "source_highlight": "fragmento",
      "target_sentence": "Oración 4 en destino",
      "target_highlight": "fragmento",
      "context_label": "Contexto 4"
    }
  ],
  "pronunciation_tip": "Tip fonético conciso (linking, flap T o entonación)"
}"""

@app.get("/")
async def read_root():
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    if os.path.exists("index.html"):
        return FileResponse("index.html", headers=headers)
    elif os.path.exists("static/index.html"):
        return FileResponse("static/index.html", headers=headers)
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
            "maxOutputTokens": 1100
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
                last_error = f"{clean_model} ({res.status_code}): {res.text}"
        except Exception as e:
            last_error = str(e)

    raise Exception(f"Error en Gemini: {last_error}")

@app.post("/api/translate")
async def translate(req: TranslationRequest):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada en Render.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío.")

    user_query = f"{SYSTEM_PROMPT}\n\nTérmino: \"{req.text}\"\nOrigen: {req.source_lang}\nDestino: {req.target_lang}"

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
