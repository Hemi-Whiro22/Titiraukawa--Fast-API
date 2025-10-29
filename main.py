# backend/main.py
import os, base64, json, aiofiles
from typing import Optional, List
from fastapi import (
    FastAPI, HTTPException, UploadFile, File, Form, Request, BackgroundTasks
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
import openai, aiohttp, asyncpg

# ───────────────────────────────────────────────
# ░░  INITIALISE  ░░
# ───────────────────────────────────────────────
app = FastAPI(title="Kitenga Backend API", version="2.0.0")

# CORS / TRUST
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
trusted_hosts = os.getenv("TRUSTED_HOSTS", "*").split(",")
if trusted_hosts != ["*"]:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

# Static
app.mount("/static", StaticFiles(directory="backend/static"), name="static")

# API Keys
openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# DB pool (optional Supabase PG direct)
DB_POOL: Optional[asyncpg.Pool] = None


# ───────────────────────────────────────────────
# ░░  MODELS  ░░
# ───────────────────────────────────────────────
class OCRPayload(BaseModel):
    image_base64: str
    @field_validator("image_base64")
    @classmethod
    def valid_b64(cls, v):
        try:
            base64.b64decode(v)
            return v
        except Exception:
            raise ValueError("Invalid base64 data")

class TranslateRequest(BaseModel):
    text: str = Field(..., max_length=8000)
    target_lang: str = Field(default="mi")  # Māori default

class SpeakRequest(BaseModel):
    text: str = Field(..., max_length=8000)

class ScribeEntry(BaseModel):
    speaker: str
    text: str
    tone: Optional[str] = "neutral"
    glyph_id: Optional[str] = "auto"
    translate: bool = False

class MemoryEntry(BaseModel):
    key: str
    value: dict


# ───────────────────────────────────────────────
# ░░  STARTUP / SHUTDOWN  ░░
# ───────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global DB_POOL
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            DB_POOL = await asyncpg.create_pool(
                dsn=os.getenv("SUPABASE_DB_URL"),
                min_size=1, max_size=5
            )
            print("🔗 Connected to Supabase PG pool.")
        except Exception as e:
            print("⚠️ PG pool failed:", e)

@app.on_event("shutdown")
async def shutdown():
    global DB_POOL
    if DB_POOL:
        await DB_POOL.close()


# ───────────────────────────────────────────────
# ░░  ROUTES  ░░
# ───────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "awake", "message": "🌕 Titiraukawa flows between realms."}

@app.get("/status")
async def status():
    """Health check for Render pings."""
    keys = {
        "openai": bool(openai.api_key),
        "supabase": bool(SUPABASE_URL and SUPABASE_KEY),
    }
    return {"status": "ok", "services": keys}


# OCR  ──────────────────────────────────────────
@app.post("/ocr")
async def ocr(payload: OCRPayload):
    """OCR using Google Vision or OpenAI fallback."""
    try:
        from google.cloud import vision
        client = vision.ImageAnnotatorClient()
        img = vision.Image(content=base64.b64decode(payload.image_base64))
        res = client.text_detection(image=img)
        if res.error.message:
            raise HTTPException(500, "Vision API error")
        text = res.text_annotations[0].description if res.text_annotations else ""
        return {"text": text or "No text found."}
    except Exception as e:
        raise HTTPException(500, f"OCR failed — {e}")


# Translate  ─────────────────────────────────────
@app.post("/translate")
async def translate(req: TranslateRequest):
    if not openai.api_key:
        raise HTTPException(503, "Missing OpenAI key")
    prompt = f"Translate to {req.target_lang}: {req.text}"
    res = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return {"translation": res.choices[0].message.content.strip()}


# Speak (TTS)  ──────────────────────────────────
@app.post("/speak")
async def speak(req: SpeakRequest):
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise HTTPException(503, "TTS key missing")
    async with aiohttp.ClientSession() as session:
        body = {
            "text": req.text,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
        }
        headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
        async with session.post(
            "https://api.elevenlabs.io/v1/text-to-speech/TxGEqnHWrfWFTfGW9XjX",
            headers=headers, json=body
        ) as resp:
            if resp.status != 200:
                raise HTTPException(500, "TTS service error")
            content = await resp.read()
            os.makedirs("backend/static", exist_ok=True)
            async with aiofiles.open("backend/static/speak.mp3", "wb") as f:
                await f.write(content)
    return {"audio_url": "/static/speak.mp3"}


# Scribe (Log + Whisper)  ───────────────────────
scribe_entries: List[dict] = []

@app.post("/scribe")
async def scribe(entry: ScribeEntry, background: BackgroundTasks):
    scribe_entries.append(entry.model_dump())
    os.makedirs("backend/static", exist_ok=True)
    async with aiofiles.open("backend/static/scribe.json", "w") as f:
        await f.write(json.dumps(scribe_entries, indent=2))

    # Generate Rongo Whisper as background task
    background.add_task(generate_whisper, entry.text)
    return {"status": "logged"}

async def generate_whisper(text: str):
    try:
        res = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"What does this mean: {text}"}],
        )
        print("Rongo Whisper →", res.choices[0].message.content[:60])
    except Exception as e:
        print("Whisper failed:", e)


# Memory Storage (Supabase vector placeholder)  ─
@app.post("/memory")
async def memory(entry: MemoryEntry):
    if DB_POOL:
        async with DB_POOL.acquire() as conn:
            await conn.execute(
                "INSERT INTO ti_memory(key, value) VALUES($1, $2)",
                entry.key, json.dumps(entry.value)
            )
    return {"stored": entry.key}


# Webhook Receiver ──────────────────────────────
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("📡 Webhook:", data)
    return {"status": "received"}


# Test Chat ─────────────────────────────────────
@app.post("/chat")
async def chat(req: TranslateRequest):
    res = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": req.text}],
        max_tokens=500,
    )
    return {"reply": res.choices[0].message.content.strip()}
# End of main.py