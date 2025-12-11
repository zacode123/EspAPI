from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
import speech_recognition as sr
from google import genai
from google.genai import types
import io, os, logging
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

# ------------------------------
# TEXT → PCM16
# ------------------------------
async def text_to_pcm16(text: str) -> bytes:
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=text)])],
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Verse"   # or 'Piper', 'Charisma', 'Studio'
                    )
                ),
                audio_format="pcm16"
            )
        )
    )
    return response.audio.data
    
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
    return "Welcome to PCM16_Speech-to-Text, Text-to-PCM16_Speech, and AI api server for ESP8266/ESP32."

@app.post("/say")
async def say_endpoint(text: str = Form(...)):
    pcm16_bytes = await text_to_pcm16(text)
    return StreamingResponse(io.BytesIO(pcm16_bytes))
    
@app.post("/hear", response_class=PlainTextResponse)
async def hear_endpoint(audio: UploadFile = File(...)):
    text, error = await audio_to_text(audio)
    if error:
        raise HTTPException(status_code=400, detail=error)
    return text

@app.post("/answer", response_class=PlainTextResponse)
async def answer_endpoint(
    question: str = Form(...),
    aimodel: str = Form("gemma-3-27b"),
    temperature: float = Form(1.0),
    max_tokens: int = Form(2048)
):
    answer = await generate_answer(question, aimodel, temperature, max_tokens)
    return answer

@app.get("/ai_say")
async def ai_say_endpoint(question: str):
    answer = await generate_answer(question, "gemini-2.5-flash-lite", 1.0, 2048)
    pcm16_bytes = await text_to_pcm16(answer)
    return StreamingResponse(io.BytesIO(pcm16_bytes), media_type="audio/L16")


@app.post("/assist")
async def assist_endpoint(audio: UploadFile = File(...),
    aimodel: str = Form("gemma-3-27b"),
    temperature: float = Form(1.0),
    max_tokens: int = Form(2048)
    ):
    question_text, error = await audio_to_text(audio)
    if error:
        return {"error": error}
    answer_text = await generate_answer(question_text, "gemini-2.0-flash", 1.0, 2048)
    pcm16_bytes = await text_to_pcm16(answer_text)
    return StreamingResponse(io.BytesIO(pcm16_bytes))
