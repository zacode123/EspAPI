from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.responses import PlainTextResponse, Response

import aiohttp
import speech_recognition as sr

from google import genai
from google.genai import types
from google.genai.errors import ServerError

import io
import os
import logging
import asyncio
import re

from dotenv import load_dotenv
from functools import lru_cache
from pydantic import BaseModel


# ------------------------------------------------
# ENV + APP
# ------------------------------------------------

load_dotenv()

app = FastAPI()

ACCESS_KEY = os.getenv("ACCESS_KEY")

# ------------------------------------------------
# DEBUG LOGGER
# ------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VOICE_API")


# ------------------------------------------------
# AUTHORIZATION (ACCESS_KEY)
# ------------------------------------------------                       

@app.middleware("http")
async def verify_access_key(request, call_next):

    if request.url.path == "/":
        return await call_next(request)

    key = request.headers.get("ACCESS_KEY")

    if not key or key != ACCESS_KEY:

        logger.warning("Unauthorized access attempt")

        return Response(
            content="Service unavailable",
            status_code=503
        )

    return await call_next(request)

# ------------------------------------------------
# GLOBAL TTS SESSION (connection pooling)
# ------------------------------------------------

tts_session = None


@app.on_event("startup")
async def startup():

    global tts_session

    timeout = aiohttp.ClientTimeout(total=20)

    tts_session = aiohttp.ClientSession(timeout=timeout)

    logger.info("TTS session started")


@app.on_event("shutdown")
async def shutdown():

    await tts_session.close()

    logger.info("TTS session closed")


# ------------------------------------------------
# GEMINI CLIENT
# ------------------------------------------------

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))



# ------------------------------------------------
# SENTENCE SPLITTER
# ------------------------------------------------

def split_sentences(text: str, max_chars=180):

    text = text.replace("\n", " ").replace("\r", " ").strip()

    parts = re.split(r'(?<=[.!?।])\s+', text)

    sentences = []
    current = ""

    for part in parts:

        part = part.strip()

        if not part:
            continue

        if len(current) + len(part) < max_chars:
            current += " " + part
        else:
            sentences.append(current.strip())
            current = part

    if current:
        sentences.append(current.strip())

    return sentences


# ------------------------------------------------
# TEXT CLEANER
# ------------------------------------------------

def filter_characters(text: str, lang: str) -> str:

    if not text:
        return ""

    if lang == "hi":
        text = re.sub(r"[A-Za-z]", "", text)

    text = re.sub(r"[*_`~#]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\([^)if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")]*\)", "", text)
    text = text.replace("“", "").replace("”", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ------------------------------------------------
# GOOGLE TTS
# ------------------------------------------------

async def fetch_tts(session, sentence, lang):

    url = "https://translate.google.com/translate_tts"

    params = {
        "ie": "UTF-8",
        "q": sentence,
        "tl": lang,
        "client": "tw-ob"
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:

        async with session.get(url, params=params, headers=headers) as resp:

            if resp.status == 200:

                audio = await resp.read()

                logger.info(f"TTS received {len(audio)} bytes")

                return audio

            return b""

    except Exception as e:

        logger.error(f"TTS exception: {e}")

        return b""


# ------------------------------------------------
# GENERATE FULL TTS (PARALLEL)
# ------------------------------------------------

async def generate_full_tts(text: str, lang: str = "en"):

    sentences = split_sentences(filter_characters(text, lang))

    tasks = [
        fetch_tts(tts_session, sentence, lang)
        for sentence in sentences
        if sentence
    ]

    results = await asyncio.gather(*tasks)

    merged = bytearray()
    first = True

    for audio in results:

        if not audio:
            continue
            
        if audio[:3] == b"ID3":
            size = (audio[6] << 21) | (audio[7] << 14) | (audio[8] << 7) | audio[9]
            audio = audio[10 + size:]
            
        if not first:
            if audio[0] == 0xFF:
                i = audio.find(b"\xff")
                if i > 0:
                    audio = audio[i:]

        merged.extend(audio)

        first = False

    return bytes(merged)

# ------------------------------------------------
# AUDIO → TEXT
# ------------------------------------------------

@lru_cache(maxsize=128)
def cached_google_recognize(audio_bytes: bytes):

    recognizer = sr.Recognizer()

    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:

        audio_content = recognizer.record(source)

        text = recognizer.recognize_google(audio_content)

        logger.info(f"Recognized: {text}")

        return text


async def audio_to_text(audio_file: UploadFile):

    try:

        audio_data = await audio_file.read()

        return cached_google_recognize(audio_data), None

    except sr.UnknownValueError:

        return "", "Could not understand audio"

    except sr.RequestError:

        raise HTTPException(status_code=500, detail="Speech recognition unavailable")


# ------------------------------------------------
# GEMINI TEXT GENERATION
# ------------------------------------------------

async def generate_answer(question: str,
                          model="gemma-3-27b-it",
                          temperature=0.7,
                          max_tokens=512):

    for attempt in range(3):

        try:

            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=question)]
                )
            ]

            config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=64,
                max_output_tokens=max_tokens,
                response_mime_type="text/plain"
            )

            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=config
            )

            return response.text

        except ServerError:

            logger.warning("Gemini busy retrying...")

            await asyncio.sleep(2)

    raise HTTPException(status_code=500, detail="AI generation failed")


# ------------------------------------------------
# ROOT
# ------------------------------------------------

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "AI Voice Assistant API fir ESP32/ESP8266 by Zahid Arman Ahmed!"


# ------------------------------------------------
# TEXT → SPEECH
# ------------------------------------------------

@app.post("/say")
async def say_endpoint(text: str = Form(...), lang: str = Form("en")):

    logger.info("/say called")

    audio = await generate_full_tts(text, lang)

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(audio)),
            "Connection": "close"
        }
    )


# ------------------------------------------------
# SPEECH → TEXT
# ------------------------------------------------

@app.post("/hear", response_class=PlainTextResponse)
async def hear_endpoint(audio: UploadFile = File(...)):

    logger.info("/hear called")

    text, error = await audio_to_text(audio)

    if error:
        raise HTTPException(status_code=400, detail=error)

    return text


# ------------------------------------------------
# AI TEXT RESPONSE
# ------------------------------------------------

@app.post("/answer", response_class=PlainTextResponse)
async def answer_endpoint(question: str = Form(...),
                          aimodel: str = Form("gemma-3-27b-it")):

    logger.info("/answer called")

    answer = await generate_answer(question, aimodel)

    return answer


# ------------------------------------------------
# AI → SPEECH REQUEST MODEL
# ------------------------------------------------

class AISayRequest(BaseModel):
    question: str


# ------------------------------------------------
# AI → SPEECH (GET)
# ------------------------------------------------

@app.get("/ai_say")
async def ai_say_get(question: str):

    logger.info("/ai_say GET called")

    answer = await generate_answer(question)

    lang = "hi" if "hindi" in question.lower() else "en"

    audio = await generate_full_tts(answer, lang)

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(audio)),
            "Connection": "close"
        }
    )


# ------------------------------------------------
# AI → SPEECH (POST)
# ------------------------------------------------

@app.post("/ai_say")
async def ai_say_post(data: AISayRequest):

    logger.info("/ai_say POST called")

    question = data.question

    answer = await generate_answer(question)

    lang = "hi" if "hindi" in question.lower() else "en"

    audio = await generate_full_tts(answer, lang)

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(audio)),
            "Connection": "close"
        }
    )


# ------------------------------------------------
# VOICE ASSISTANT
# ------------------------------------------------

@app.post("/assist")
async def assist_endpoint(audio: UploadFile = File(...),
                          aimodel: str = Form("gemma-3-12b-it")):

    logger.info("/assist called")

    question_text, error = await audio_to_text(audio)

    if error:
        raise HTTPException(status_code=400, detail=error)

    answer_text = await generate_answer(question_text, aimodel)

    audio = await generate_full_tts(answer_text)

    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={
            "Content-Length": str(len(audio)),
            "Connection": "close"
        }
    )
