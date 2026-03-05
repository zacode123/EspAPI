from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse, Response

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

    logger.info("Splitting sentences")

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

    logger.info(f"Total sentences: {len(sentences)}")

    for i, s in enumerate(sentences):
        logger.info(f"Sentence {i+1}: {s}")

    return sentences


# ------------------------------------------------
# TEXT CLEANER
# ------------------------------------------------

def filter_characters(text: str, lan: str) -> str:
    if not text:
        return ""

    original = text
    
    if lang == "hi":
        text = re.sub(r"[^\u0900-\u097F\s.,!?।]", "", text)
        
    text = re.sub(r"[*_`~#]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.replace("“", "").replace("”", "")
    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    if original != text:
        logger.debug(f"Cleaned text: {text}")

    return text


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

    logger.info(f"TTS request: {sentence}")

    try:

        async with session.get(url, params=params, headers=headers) as resp:

            logger.info(f"TTS HTTP status: {resp.status}")

            if resp.status == 200:

                audio = await resp.read()

                logger.info(f"Received audio bytes: {len(audio)}")

                return audio

            else:

                logger.warning("TTS failed")

                return b""

    except Exception as e:

        logger.error(f"TTS exception: {e}")

        return b""


# ------------------------------------------------
# STREAM TTS
# ------------------------------------------------

async def stream_tts(text: str, lang: str = "en"):

    logger.info("===== STREAM START =====")

    sentences = split_sentences(text)

    connector = aiohttp.TCPConnector(limit=2)

    first_chunk = True

    async with aiohttp.ClientSession(connector=connector) as session:

        for i, sentence in enumerate(sentences):

            sentence = filter_characters(sentence)

            if not sentence:
                continue

            logger.info(f"Processing sentence {i+1}")

            audio = await fetch_tts(session, sentence, lang)

            if not audio:
                continue

            if first_chunk:

                logger.info("Sending FIRST chunk")

                first_chunk = False

                yield audio

            else:

                # Remove MP3 header to keep browser happy
                logger.info("Sending FOLLOWUP chunk")

                yield audio[500:]

            await asyncio.sleep(0.01)

    logger.info("===== STREAM END =====")


# ------------------------------------------------
# AUDIO → TEXT
# ------------------------------------------------

@lru_cache(maxsize=128)
def cached_google_recognize(audio_bytes: bytes):

    logger.info("Speech recognition start")

    recognizer = sr.Recognizer()

    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:

        audio_content = recognizer.record(source)

        text = recognizer.recognize_google(audio_content)

        logger.info(f"Recognized: {text}")

        return text


async def audio_to_text(audio_file: UploadFile):

    try:

        audio_data = await audio_file.read()

        logger.info(f"Audio bytes received: {len(audio_data)}")

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

    logger.info(f"Generating answer for: {question}")

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

            logger.info("Gemini response generated")

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

    logger.info("Endpoint /say called")

    return StreamingResponse(
        stream_tts(text, lang),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked"
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
# AI → SPEECH
# ------------------------------------------------

@app.get("/ai_say")
async def ai_say_endpoint(question: str):

    logger.info("/ai_say called")

    answer = await generate_answer(question)

    lang = "hi" if "hindi" in question.lower() else "en"

    return StreamingResponse(
        stream_tts(answer, lang),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked"
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

    return StreamingResponse(
        stream_tts(answer_text),
        media_type="audio/mpeg",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked"
        }
    )
