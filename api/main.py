from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
import aiohttp
import speech_recognition as sr
from google import genai
from google.genai import types
import io, os, logging, asyncio
from dotenv import load_dotenv
from functools import lru_cache
from typing import List

load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔑 Gemini API
@lru_cache(maxsize=1)
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("❌ GEMINI_API_KEY missing!")
        raise RuntimeError("Missing GEMINI_API_KEY in environment")
    return genai.Client(api_key=api_key)

client = get_gemini_client()
logger.info("✅ Gemini AI initialized")

# ------------------------------
# TEXT → MP3
# ------------------------------
async def text_to_mp3(text: str, lang: str = "en") -> bytes:
    MAX_CHARS = 200
    chunks = []
    t = text.strip()
    while t:
        chunk = t[:MAX_CHARS]
        if len(chunk) == MAX_CHARS and " " in chunk:
            last_space = chunk.rfind(" ")
            chunk, t = chunk[:last_space], t[last_space+1:]
        else:
            t = t[MAX_CHARS:]
        chunks.append(chunk.strip())
    url = "https://translate.google.com/translate_tts"
    headers = {"User-Agent": "Mozilla/5.0"}
    async def fetch_chunk(session, chunk_text):
        params = {
            "ie": "UTF-8",
            "q": chunk_text,
            "tl": lang,
            "client": "tw-ob"
        }
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                raise RuntimeError(f"TTS request failed for chunk: {resp.status}")
    async with aiohttp.ClientSession() as session:
        audio_parts = await asyncio.gather(*(fetch_chunk(session, c) for c in chunks))
    return b"".join(audio_parts)

# ------------------------------
# AUDIO → TEXT
# ------------------------------
@lru_cache(maxsize=128)
def cached_google_recognize(audio_bytes: bytes) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
        audio_content = recognizer.record(source)
        return recognizer.recognize_google(audio_content)

async def audio_to_text(audio_file: UploadFile):
    try:
        audio_data = await audio_file.read()
        return cached_google_recognize(audio_data), None
    except sr.UnknownValueError:
        return "", "Could not understand audio"
    except sr.RequestError:
        raise HTTPException(status_code=500, detail="Speech recognition service unavailable")
    except Exception:
        raise HTTPException(status_code=500, detail="Error processing audio")

# ------------------------------
# GEMINI AI TEXT GENERATION
# ------------------------------
@lru_cache(maxsize=256)
def generate_response(question: str, aimodel: str, temperature: float, max_tokens: int) -> str:
    model = aimodel
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature, top_p=0.95, top_k=64,
        max_output_tokens=max_tokens, response_mime_type="text/plain"
    )
    response = client.models.generate_content(
        model=model, contents=contents, config=generate_content_config
    )
    return response.text

async def generate_answer(question: str, aimodel: str, temperature: float = 1.0, max_tokens: int = 2048):
    try:
        return generate_response(question, aimodel, temperature, max_tokens)
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")

# ------------------------------
# ROUTES
# ------------------------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Welcome to MP3_Speech-to-Text, Text-to-MP3_Speech, and AI API server for ESP8266/ESP32."

@app.post("/say")
async def say_endpoint(text: str = Form(...), lang: str = Form("en")):
    mp3_bytes = await text_to_mp3(text, lang)
    return StreamingResponse(io.BytesIO(mp3_bytes), media_type="audio/mpeg")

@app.post("/hear", response_class=PlainTextResponse)
async def hear_endpoint(audio: UploadFile = File(...)):
    text, error = await audio_to_text(audio)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return text

@app.post("/answer", response_class=PlainTextResponse)
async def answer_endpoint(
    question: str = Form(...),
    aimodel: str = Form("gemma-3-27b-it"),
    temperature: float = Form(1.0),
    max_tokens: int = Form(2048)
):
    answer = await generate_answer(question, aimodel, temperature, max_tokens)
    return answer

@app.get("/ai_say")
async def ai_say_endpoint(question: str):
    answer = await generate_answer(question, "gemma-3-27b-it", 1.0, 2000)
    lang = "hi" if "hindi" in answer.lower() else "en"
    mp3_bytes = await text_to_mp3(answer, lang)
    return StreamingResponse(io.BytesIO(mp3_bytes), media_type="audio/mpeg")
    
@app.post("/assist")
async def assist_endpoint(
    audio: UploadFile = File(...),
    aimodel: str = Form("gemini-2.5-flash-lite"),
    temperature: float = Form(1.0),
    max_tokens: int = Form(2048)
):
    question_text, error = await audio_to_text(audio)
    if error:
        return {"error": error}
    answer_text = await generate_answer(question_text, aimodel, temperature, max_tokens)
    mp3_bytes = await text_to_mp3(answer_text)
    return StreamingResponse(io.BytesIO(mp3_bytes), media_type="audio/mpeg")
