import os
import re
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
Tu misión es entregar un corpus bilingüe con ejemplos reales y traducciones equivalentes.

IMPORTANTE: Responde ÚNICAMENTE con un JSON válido y bien formateado con comillas dobles estándar en las propiedades y valores.
Estructura exacta:
{
  "query": "término buscado",
  "main_translation": "Traducción principal más natural",
  "part_of_speech": "Tipo de palabra (Verb, Phrasal Verb, Noun, Idiom, Adj)",
  "ipa_us": "Transcripción IPA en inglés americano",
  "ipa_uk": "Transcripción IPA en inglés británico",
  "category_topic": "Temática general (ej: Daily Life, Relationships, Business)",
  "translations_chips": [
    "traducción 1", "traducción 2", "traducción 3", "traducción 4", "traducción 5", "traducción 6"
  ],
  "context_examples": [
    {
      "source_sentence": "Oración de ejemplo en idioma origen",
      "source_highlight": "fragmento clave a resaltar",
      "target_sentence": "Oración traducida al idioma destino",
      "target_highlight": "fragmento traducido clave",
      "context_label": "Contexto de la oración (ej: Coloquial, Diario, Pregunta)"
    },
    {
      "source_sentence": "Oración 2 en origen",
      "source_highlight": "fragmento 2",
      "target_sentence": "Oración 2 en destino",
      "target_highlight": "fragmento 2",
      "context_label": "Contexto 2"
    },
    {
      "source_sentence": "Oración 3 en origen",
      "source_highlight": "fragmento 3",
      "target_sentence": "Oración 3 en destino",
      "target_highlight": "fragmento 3",
      "context_label": "Contexto 3"
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

def clean_and_parse_json(raw_text: str) -> dict:
    """Extrae y repara el JSON de forma robusta frente a anomalías de formato."""
    text = raw_text.strip()
    
    # 1. Limpiar bloques de código Markdown
    text = re.sub(r"^
http://googleusercontent.com/immersive_entry_chip/0

Haz clic en **Commit changes...**.

---

### Paso 2: Probar la aplicación

1. Espera unos 30 segundos a que Render actualice el despliegue a **Live**.
2. Abre tu enlace: **`https://context-translator.onrender.com`**
3. Presiona **`Ctrl + Shift + R`** para asegurar la carga limpia.
4. Escribe cualquier término (por ejemplo: `que paso?`, `get along` o `break down`) y pulsa **Buscar**.

El sistema procesará la respuesta sin fallos y verás el panel de Reverso Context con las píldoras de traducción y los ejemplos bilingües resaltados.
