from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from gtts import gTTS
import speech_recognition as sr
from google import genai
from google.genai import types
import io, os, asyncio, logging, audioop, miniaudio, wave
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from functools import lru_cache

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

executor = ThreadPoolExecutor(max_workers=5)

# ------------------------------
# TEXT → WAV
# ------------------------------
async def text_to_wav(text: str) -> bytes:
    def synthesize_speech():
        mp3_io = io.BytesIO()
        gTTS(text=text, lang="en").write_to_fp(mp3_io)
        mp3_io.seek(0)
        dec = miniaudio.decode(mp3_io.read(), filetype="mp3")
        pcm = dec.samples
        sr = dec.sample_rate
        ch = dec.nchannels
        width = dec.sample_width
        if ch > 1:
            pcm = audioop.tomono(pcm, width, 0.5, 0.5)
        if sr != 16000 or ch != 1:
            pcm = audioop.ratecv(pcm, width, ch, sr, 16000, None)[0]
            sr = 16000
            ch = 1
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setnchannels(ch)
            wf.setsampwidth(width)
            wf.setframerate(sr)
            wf.writeframes(pcm)
        wav_io.seek(0)
        return wav_io.read()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, synthesize_speech)

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
def cached_generate_answer(question: str, temperature: float, max_tokens: int) -> str:
    model = "gemini-2.0-flash"
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature, top_p=0.95, top_k=64,
        max_output_tokens=max_tokens, response_mime_type="text/plain"
    )
    response = client.models.generate_content(
        model=model, contents=contents, config=generate_content_config
    )
    return response.text

async def generate_answer(question: str, temperature: float = 1.0, max_tokens: int = 2048):
    try:
        return cached_generate_answer(question, temperature, max_tokens)
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")

# ------------------------------
# ROUTES
# ------------------------------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Welcome to PCM Speech-to-Text, Text-to-Speech, and AI assistant!"

@app.post("/say")
async def say_endpoint(text: str = Form(...)):
    wav_bytes = await text_to_wav(text)
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=speech.wav"})
    
@app.post("/hear", response_class=PlainTextResponse)
async def hear_endpoint(audio: UploadFile = File(...)):
    text, error = await audio_to_text(audio)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return text

@app.post("/answer", response_class=PlainTextResponse)
async def answer_endpoint(
    question: str = Form(...),
    temperature: float = Form(1.0),
    max_tokens: int = Form(2048),
):
    answer = await generate_answer(question, temperature, max_tokens)
    return answer

@app.post("/assist")
async def assist_endpoint(audio: UploadFile = File(...)):
    question_text, error = await audio_to_text(audio)
    if error:
        return {"error": error}
    answer_text = await generate_answer(question_text, temperature=1.0, max_tokens=2048)
    wav_bytes = await text_to_wav(answer_text)
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=speech.wav"})
  
