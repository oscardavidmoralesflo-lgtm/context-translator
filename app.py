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

def get_best_available_model():
    """Detecta automáticamente el mejor modelo activo en la cuenta."""
    try:
        available_models = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        
        # 1. Priorizar modelos Flash
        for m in available_models:
            if "flash" in m.lower():
                return m
                
        # 2. Priorizar cualquier modelo Gemini
        for m in available_models:
            if "gemini" in m.lower():
                return m
                
        # 3. Fallback al primer modelo con soporte para generación
        if available_models:
            return available_models[0]
            
    except Exception as e:
        print(f"Error listando modelos: {e}")
        
    return "models/gemini-1.5-flash"

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
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada en Render.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Texto vacío.")

    try:
        genai.configure(api_key=api_key)
        selected_model = get_best_available_model()

        model = genai.GenerativeModel(
            model_name=selected_model,
            generation_config={"response_mime_type": "application/json"}
        )

        prompt = f"""{SYSTEM_PROMPT}

Texto a traducir/analizar: "{req.text}"
Idioma origen: {req.source_lang}
Idioma destino: {req.target_lang}
Tono/Énfasis: {req.tone_preference}"""

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
        print("=== ERROR DETALLADO ===")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error en traducción ({type(e).__name__}): {str(e)}")

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
