from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from gtts import gTTS
import speech_recognition as sr
from google import genai
from google.genai import types
import io, os, asyncio, logging
from pymp3 import Decoder
import numpy as np
import resampy
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("❌ GEMINI_API_KEY missing!")
    raise RuntimeError("Missing GEMINI_API_KEY in environment")

client = genai.Client(api_key=GEMINI_API_KEY)
logger.info("✅ Gemini AI initialized")

executor = ThreadPoolExecutor(max_workers=5)

async def text_to_pcm(text: str):
    def synthesize_speech():
        mp3_io = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(mp3_io)
        mp3_io.seek(0)
        decoder = Decoder(mp3_io.read())
        pcm = decoder.decode()
        samples = pcm.samples
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        resampled = resampy.resample(samples, pcm.sample_rate, 16000)
        pcm16 = np.int16(resampled * 32767).tobytes()
        return pcm16
    return await asyncio.get_running_loop().run_in_executor(executor, synthesize_speech)

async def audio_to_text(audio_file: UploadFile):
    try:
        audio_data = await audio_file.read()
        audio_io = io.BytesIO(audio_data)
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_io) as source:
            audio_content = recognizer.record(source)
            text = recognizer.recognize_google(audio_content)
        return text, None
    except sr.UnknownValueError:
        return "", "Could not understand audio"
    except sr.RequestError:
        raise HTTPException(status_code=500, detail="Speech recognition service unavailable")
    except Exception as e:
        logger.error(f"❌ Audio processing error: {e}")
        raise HTTPException(status_code=500, detail="Error processing audio")

async def generate_answer(question: str, temperature: float = 1.0, max_tokens: int = 2048):
    try:
        model = "gemini-2.0-flash"
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        config = types.GenerateContentConfig(
            temperature=temperature, top_p=0.95, top_k=40,
            max_output_tokens=max_tokens, response_mime_type="text/plain"
        )
        response = client.models.generate_content(model=model, contents=contents, config=config)
        return response.text if hasattr(response, "text") else ""
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Speech ↔ Text ↔ AI Assistant Ready ✅"

@app.post("/say")
async def say_endpoint(text: str = Form(...)):
    pcm_bytes = await text_to_pcm(text)
    return StreamingResponse(io.BytesIO(pcm_bytes), media_type="application/octet-stream")

@app.post("/hear")
async def hear_endpoint(audio: UploadFile = File(...)):
    text, error = await audio_to_text(audio)
    if error:
        return {"error": error}
    return text

@app.post("/answer")
async def answer_endpoint(question: str = Form(...), temperature: float = Form(1.0), max_tokens: int = Form(2048)):
    return await generate_answer(question, temperature, max_tokens)

@app.post("/assist")
async def assist_endpoint(audio: UploadFile = File(...)):
    question_text, error = await audio_to_text(audio)
    if error:
        return {"error": error}
    answer_text = await generate_answer(question_text, temperature=1.0, max_tokens=2048)
    pcm_bytes = await text_to_pcm(answer_text)
    return StreamingResponse(io.BytesIO(pcm_bytes), media_type="application/octet-stream")
