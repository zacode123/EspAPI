from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse, Response
import aiohttp
import speech_recognition as sr
from google import genai
from google.genai import types
import io, os, logging, asyncio, re
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------
# GEMINI CLIENT
# ------------------------------

@lru_cache(maxsize=1)
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    return genai.Client(api_key=api_key)

client = get_gemini_client()

# ------------------------------
# SENTENCE SPLITTER
# ------------------------------

def split_sentences(text: str, max_chars=180):
    text = text.replace("\n", " ").replace("\r", " ").strip()

    sentences = []
    current = ""

    for part in re.split(r"[.!?]+", text):
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

# ------------------------------
# TTS CHUNK STREAM
# ------------------------------

async def stream_tts(text: str, lang: str = "en"):
    logger.info(f"🔹 Starting TTS stream | lang={lang}")
    sentences = split_sentences(text)
    logger.info(f"🔹 Total sentences to process: {len(sentences)}")

    url = "https://translate.google.com/translate_tts"
    headers = {"User-Agent": "Mozilla/5.0", "Connection": "keep-alive"}
    connector = aiohttp.TCPConnector(limit=5, ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:

        for idx, sentence in enumerate(sentences, start=1):
            sentence = sentence.strip()
            if not sentence:
                logger.debug(f"⚠️ Skipping empty sentence #{idx}")
                continue

            params = {"ie": "UTF-8", "q": sentence, "tl": lang, "client": "tw-ob"}
            logger.info(f"🔹 Sending TTS request for sentence #{idx}: {sentence[:50]}...")

            try:
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        audio = await resp.read()
                        logger.info(f"✅ Received audio for sentence #{idx} ({len(audio)} bytes)")
                        yield audio
                    else:
                        logger.warning(f"❌ TTS failed ({resp.status}) for sentence #{idx}: {sentence[:50]}...")
            except Exception as e:
                logger.error(f"💥 Exception during TTS for sentence #{idx}: {e}")
                
# ------------------------------
# AUDIO → TEXT
# ------------------------------

@lru_cache(maxsize=128)
def cached_google_recognize(audio_bytes: bytes):

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
        raise HTTPException(status_code=500, detail="Speech recognition unavailable")

# ------------------------------
# GEMINI TEXT GENERATION
# ------------------------------

async def generate_answer(question: str,
                          model="gemma-3-27b-it",
                          temperature=1.0,
                          max_tokens=2048):

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

# ------------------------------
# ROUTES
# ------------------------------

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "AI Voice Assistant API"

@app.get("/favicon.ico")
@app.get("/favicon.png")
async def favicon():
    return Response(status_code=204)

# TEXT → SPEECH

@app.post("/say")
async def say_endpoint(text: str = Form(...), lang: str = Form("en")):
    return StreamingResponse(stream_tts(text, lang), media_type="audio/mpeg")

# SPEECH → TEXT

@app.post("/hear", response_class=PlainTextResponse)
async def hear_endpoint(audio: UploadFile = File(...)):

    text, error = await audio_to_text(audio)

    if error:
        raise HTTPException(status_code=400, detail=error)

    return text

# AI TEXT RESPONSE

@app.post("/answer", response_class=PlainTextResponse)
async def answer_endpoint(
        question: str = Form(...),
        aimodel: str = Form("gemma-3-27b-it")):

    answer = await generate_answer(question, aimodel)
    return answer

# AI → SPEECH (REALTIME)

@app.get("/ai_say")
async def ai_say_endpoint(question: str):

    answer = await generate_answer(question)

    lang = "hi" if "hindi" in question.lower() else "en"

    return StreamingResponse(
        stream_tts(answer, lang),
        media_type="audio/mpeg"
    )

# VOICE ASSISTANT

@app.post("/assist")
async def assist_endpoint(
        audio: UploadFile = File(...),
        aimodel: str = Form("gemma-3-27b-it")):

    question_text, error = await audio_to_text(audio)

    if error:
        raise HTTPException(status_code=400, detail=error)

    answer_text = await generate_answer(question_text, aimodel)

    return StreamingResponse(
        stream_tts(answer_text),
        media_type="audio/mpeg"
    )
