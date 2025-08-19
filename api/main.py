from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from google.cloud import texttospeech
import speech_recognition as sr
from google import genai
from google.genai import types
import io, os, asyncio, logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment
load_dotenv()
app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("❌ GEMINI_API_KEY missing!")
    raise RuntimeError("Missing GEMINI_API_KEY in environment")
client = genai.Client(api_key=GEMINI_API_KEY)
logger.info("✅ Gemini AI initialized")

# Google Cloud TTS
tts_client = texttospeech.TextToSpeechClient()
executor = ThreadPoolExecutor(max_workers=5)


# ---------- TTS (returns raw PCM16) ----------
async def text_to_pcm(text: str) -> bytes:
    def synthesize():
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="en-US")
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )
        response = tts_client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        return response.audio_content  # raw PCM16
    return await asyncio.get_running_loop().run_in_executor(executor, synthesize)


# ---------- STT ----------
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
    except Exception:
        raise HTTPException(status_code=500, detail="Error processing audio")


# ---------- AI Answer ----------
async def generate_answer(question: str, temperature: float = 1.0, max_tokens: int = 8192):
    try:
        model = "gemini-2.0-flash"
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        config = types.GenerateContentConfig(
            temperature=temperature, top_p=0.95, top_k=64, max_output_tokens=max_tokens,
            response_mime_type="text/plain"
        )
        response = ""
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
            if chunk.text:
                response += chunk.text
        return response
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")


# ---------- Endpoints ----------
@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Welcome to PCM Speech-to-Text, Text-to-Speech, and AI assistant!"

@app.post("/say")
async def say_endpoint(text: str = Form(...)):
    pcm_bytes = await text_to_pcm(text)
    return StreamingResponse(io.BytesIO(pcm_bytes),
                             media_type="audio/raw",  # raw PCM16 stream
                             headers={"Content-Disposition": "attachment; filename=speech.pcm"})

@app.post("/hear")
async def hear_endpoint(audio: UploadFile = File(...)):
    text, error = await audio_to_text(audio)
    if error:
        return {"error": error}
    return text

@app.post("/answer")
async def answer_endpoint(question: str = Form(...), temperature: float = Form(1.0), max_tokens: int = Form(8192)):
    answer = await generate_answer(question, temperature, max_tokens)
    return answer

@app.post("/assist")
async def assist_endpoint(audio: UploadFile = File(...)):
    question_text, error = await audio_to_text(audio)
    if error:
        return {"error": error}
    answer_text = await generate_answer(question_text, temperature=1.0, max_tokens=8192)
    pcm_bytes = await text_to_pcm(answer_text)
    return StreamingResponse(io.BytesIO(pcm_bytes),
                             media_type="audio/raw",
                             headers={"Content-Disposition": "attachment; filename=answer.pcm"})
