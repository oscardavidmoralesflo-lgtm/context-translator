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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
Eres un motor de traducción contextual y lingüista de élite. Proporciona traducciones precisas junto con un análisis profundo del contexto cultural, tono, matices y transcripción fonética IPA.
Responde ÚNICAMENTE con un JSON válido con esta estructura:
{
  "main_translation": "Traducción principal más natural y precisa",
  "ipa_target": "Transcripción fonética IPA",
  "tone_detected": "Tono (ej. Formal, Coloquial, Corporativo)",
  "alternatives": [
    {
      "text": "Frase alternativa",
      "context": "Cuándo usar esta opción",
      "register": "Formal / Informal / Slang"
    }
  ],
  "cultural_nuances": [
    "Explicación de modismos o notas culturales relevantes"
  ]
}
"""

@app.get("/")
async def read_root():
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h2>Error: No se encontró el archivo index.html en el repositorio.</h2>")

@app.post("/api/translate")
async def translate(req: TranslationRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada.")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío.")

    try:
        model = genai.GenerativeModel("gemini-1.5-flash", generation_config={"response_mime_type": "application/json"})
        prompt = f"{SYSTEM_PROMPT}\nTexto: \"{req.text}\"\nOrigen: {req.source_lang}\nDestino: {req.target_lang}\nTono: {req.tone_preference}"
        response = await asyncio.to_thread(model.generate_content, prompt)
        return json.loads(response.text)
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

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
