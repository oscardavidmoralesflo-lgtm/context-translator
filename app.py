import os
import json
import asyncio
import edge_tts
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
Eres un lingüista y motor de traducción contextual avanzado. Tu objetivo es desglosar de forma exhaustiva TODOS los posibles significados y usos de la palabra o frase según cada contexto.

Debes responder EXCLUSIVAMENTE con un JSON válido con esta estructura:
{
  "main_translation": "Traducción más común y natural",
  "ipa_target": "Transcripción fonética IPA de la traducción principal",
  "all_meanings": [
    {
      "context_category": "Coloquial / Diario",
      "translation": "Traducción específica para este contexto",
      "example_usage": "Ejemplo en una frase corta",
      "nuance": "Cuándo y por qué se usa aquí"
    },
    {
      "context_category": "Formal / Profesional",
      "translation": "Traducción formal",
      "example_usage": "Ejemplo formal",
      "nuance": "Uso en correos, negocios o reuniones"
    },
    {
      "context_category": "Slang / Modismo / Jerga",
      "translation": "Traducción informal / modismo",
      "example_usage": "Ejemplo informal",
      "nuance": "Matiz cultural o regional"
    },
    {
      "context_category": "Otros significados / Acepciones",
      "translation": "Significado alternativo o secundario",
      "example_usage": "Ejemplo de uso",
      "nuance": "Significados secundarios o dobles sentidos"
    }
  ],
  "pronunciation_tip": "Consejo clave sobre fonética, entonación o enlace de sonidos"
}
"""

@app.get("/")
async def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    elif os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return HTMLResponse("<h2>Error: No se encontró index.html</h2>")

@app.post("/api/translate")
async def translate(req: TranslationRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ERROR: La variable GEMINI_API_KEY no está configurada en Render.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json"})
        
        prompt = f"""
        {SYSTEM_PROMPT}

        Texto/Palabra a analizar: "{req.text}"
        Idioma de origen: {req.source_lang}
        Idioma de destino: {req.target_lang}
        Tono preferido: {req.tone_preference}
        """

        response = await asyncio.to_thread(model.generate_content, prompt)
        return json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en Gemini API: {str(e)}")

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
