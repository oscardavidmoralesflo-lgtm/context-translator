import os
import json
import asyncio
import traceback
import edge_tts
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

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
Responde ÚNICAMENTE con un JSON válido con esta estructura exacta (sin texto introductorio ni bloques de código extra):
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

# Lista de modelos compatibles en orden de preferencia
CANDIDATE_MODELS = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro-latest",
    "gemini-pro"
]

@app.get("/")
async def read_root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    elif os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return HTMLResponse("<h2>Error: No se encontró index.html</h2>")

@app.post("/api/translate")
async def translate(req: TranslationRequest):
    raw_key = os.getenv("GEMINI_API_KEY", "")
    api_key = raw_key.strip()

    if not api_key:
        raise HTTPException(status_code=500, detail="Variable GEMINI_API_KEY no configurada en Render.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Texto vacío.")

    genai.configure(api_key=api_key)

    prompt = f"""{SYSTEM_PROMPT}

Texto a traducir/analizar: "{req.text}"
Idioma origen: {req.source_lang}
Idioma destino: {req.target_lang}
Tono/Énfasis: {req.tone_preference}"""

    last_error = None
    
    # Intentar con la lista de modelos disponibles
    for model_name in CANDIDATE_MODELS:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={"response_mime_type": "application/json"}
            )
            response = await asyncio.to_thread(model.generate_content, prompt)
            raw_text = response.text.strip()

            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            return json.loads(raw_text.strip())
        except Exception as e:
            last_error = e
            continue

    print("=== ERROR DETALLADO ===")
    print(traceback.format_exc())
    raise HTTPException(status_code=500, detail=f"No se pudo conectar con ningún modelo Gemini: {str(last_error)}")

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
