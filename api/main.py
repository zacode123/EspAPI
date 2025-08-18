from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from gtts import gTTS
import speech_recognition as sr
from google import genai
from google.genai import types
import io, os, asyncio, logging
from pydub import AudioSegment
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

async def text_to_wav(text: str):
    def synthesize_speech():
        tts = gTTS(text=text, lang='en')
        mp3_io = io.BytesIO()
        tts.write_to_fp(mp3_io)
        mp3_io.seek(0)
        audio = AudioSegment.from_file(mp3_io, format="mp3")
        wav_io = io.BytesIO()
        audio.set_frame_rate(16000).set_channels(1).set_sample_width(2).export(wav_io, format="wav")
        wav_io.seek(0)
        return wav_io
    wav_io = await asyncio.get_running_loop().run_in_executor(executor, synthesize_speech)
    return wav_io

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

async def generate_answer(question: str, temperature: float = 1.0, max_tokens: int = 8192):
    try:
        model = "gemini-2.0-flash"
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature, top_p=0.95, top_k=64, max_output_tokens=max_tokens, response_mime_type="text/plain"
        )
        response = ""
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
            if chunk.text:
                response += chunk.text
        return response
    except Exception as e:
        logger.error(f"❌ Error generating answer: {e}")
        raise HTTPException(status_code=500, detail="Error generating response")

@app.get("/", response_class=PlainTextResponse)
async def root():
    return "Welcome to Speech-to-Text, Text-to-Speech, and AI-powered assistant!"

@app.post("/say")
async def say_endpoint(text: str = Form(...)):
    """Text-to-speech endpoint."""
    wav_io = await text_to_wav(text)
    return StreamingResponse(wav_io, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=speech.wav"})

@app.post("/hear")
async def hear_endpoint(audio: UploadFile = File(...)):
    text, error = await audio_to_text(audio)
    if error:
        return {"text": "", "error": error}
    return {"text": text}

@app.post("/answer")
async def answer_endpoint(question: str = Form(...), temperature: float = Form(1.0), max_tokens: int = Form(8192)):
    answer = await generate_answer(question, temperature, max_tokens)
    return {"answer": answer}

@app.post("/assist")
async def assist_endpoint(audio: UploadFile = File(...)):
    question_text, error = await audio_to_text(audio)
    if error:
        return {"text": "", "error": error}
    answer_text = await generate_answer(question_text, temperature=1.0, max_tokens=8192)
    wav_io = await text_to_wav(answer_text)
    return StreamingResponse(wav_io, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=answer.wav"})
