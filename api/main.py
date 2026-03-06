from fastapi import FastAPI, File, UploadFile, Form, HTTPException
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


load_dotenv()

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VOICE_API")


# ------------------------------------------------
# GEMINI CLIENT
# ------------------------------------------------

@lru_cache(maxsize=1)
def get_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    logger.info("Gemini client initialized")

    return genai.Client(api_key=api_key)


client = get_gemini_client()


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
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("“", "").replace("”", "")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ------------------------------------------------
# GOOGLE TTS REQUEST
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
# GENERATE FULL TTS AUDIO
# ------------------------------------------------

async def generate_full_tts(text: str, lang: str = "en"):

    sentences = split_sentences(filter_characters(text, lang))

    audio_bytes = bytearray()

    async with aiohttp.ClientSession() as session:

        for sentence in sentences:

            if not sentence:
                continue

            audio = await fetch_tts(session, sentence, lang)

            if not audio:
                continue

            audio_bytes.extend(audio)

    return bytes(audio_bytes)


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
                          temperature=1.0,
                          max_tokens=2048):

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

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )

            return response.text

        except ServerError:

            logger.warning("Gemini busy retrying...")

            await asyncio.sleep(3)

    raise HTTPException(status_code=500, detail="AI generation failed")


# ------------------------------------------------
# ROUTES
# ------------------------------------------------

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "AI Voice Assistant API"


@app.get("/favicon.ico")
@app.get("/favicon.png")
async def favicon():
    return Response(status_code=204)


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
        headers={"Content-Length": str(len(audio))}
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
# AI → SPEECH
# ------------------------------------------------

@app.get("/ai_say")
async def ai_say_endpoint(question: str):

    logger.info("/ai_say called")

    answer = await generate_answer(question)

    lang = "hi" if "hindi" in question.lower() else "en"

    audio = await generate_full_tts(answer, lang)

    return Response(
        content=audio,
        media_type="audio/mpeg"
     #   headers={"Content-Length": str(len(audio))}
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
        headers={"Content-Length": str(len(audio))}
    )
