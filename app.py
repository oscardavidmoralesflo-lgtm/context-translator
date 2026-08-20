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

def get_dynamic_model(api_key: str):
    """Consulta directamente a Google qué modelos están activos para tu cuenta."""
    headers = {"x-goog-api-key": api_key}
    for version in ["v1beta", "v1"]:
        try:
            url = f"https://generativelanguage.googleapis.com/{version}/models"
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                data = res.json()
                models = data.get("models", [])
                
                # Filtrar modelos que soporten generación de contenido
                supported = [
                    m["name"] for m in models 
                    if "generateContent" in m.get("supportedGenerationMethods", [])
                ]
                
                # Prioridad 1: Modelos Flash
                for m in supported:
                    if "flash" in m.lower():
                        return version, m
                
                # Prioridad 2: Cualquier modelo Gemini disponible
                if supported:
                    return version, supported[0]
            else:
                print(f"Error listando modelos en {version}: {res.status_code} {res.text}", flush=True)
        except Exception as e:
            print(f"Excepción consultando {version}: {e}", flush=True)
            
    return "v1beta", "models/gemini-1.5-flash"

@app.post("/api/translate")
async def translate(req: TranslationRequest):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY no configurada en Render.")

    if not req.text.strip():
        raise HTTPException(status_code=400, detail="El texto está vacío.")

    api_version, model_name = get_dynamic_model(api_key)
    
    # model_name ya incluye el prefijo 'models/'
    url = f"https://generativelanguage.googleapis.com/{api_version}/{model_name}:generateContent"
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
            print(f"Fallo en llamada a Gemini ({url}): {res.status_code} - {res.text}", flush=True)
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
        raise HTTPException(status_code=500, detail="El modelo no devolvió un JSON válido.")
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
