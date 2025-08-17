from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, PlainTextResponse
from gtts import gTTS
import speech_recognition as sr
from google import genai
from google.genai import types
import io
import os
import asyncio
import logging
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("❌ ERROR: GEMINI_API_KEY is missing!")
    raise RuntimeError("Missing GEMINI_API_KEY in environment")

# Initialize Gemini AI client
client = genai.Client(api_key=GEMINI_API_KEY)
logger.info("✅ Gemini AI initialized successfully")

# ThreadPool for CPU-heavy tasks
executor = ThreadPoolExecutor(max_workers=5)

async def generate_answer(question: str, temperature: float, max_tokens: int):
    """Generate an AI response using Gemini API."""
    try:
        model = "gemini-2.0-pro-exp-02-05"
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
async def text_to_speech(text: str = Form(...)):
    """Convert text to speech and return as a WAV file."""
    logger.info(f"📢 /say request received. Text length: {len(text)} characters")

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

    logger.info(f"🎶 Converted text to WAV ({len(wav_io.getvalue())} bytes)")

    return StreamingResponse(wav_io, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=speech.wav"})

@app.post("/hear")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech to text."""
    logger.info(f"👂 /hear request received. File: {audio.filename} ({audio.content_type})")

    try:
        audio_data = await audio.read()
        audio_io = io.BytesIO(audio_data)

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_io) as source:
            audio_content = recognizer.record(source)
            text = recognizer.recognize_google(audio_content)
        
        logger.info(f"📝 Recognized text: {text[:100]}...")
        return {"text": text}

    except sr.UnknownValueError:
        logger.warning("🤷 Could not understand audio")
        return {"text": "", "error": "Could not understand audio"}

    except sr.RequestError as e:
        logger.error(f"❌ Google Speech Recognition error: {e}")
        raise HTTPException(status_code=500, detail="Speech recognition service unavailable")

    except Exception as e:
        logger.error(f"❌ /hear error: {e}")
        raise HTTPException(status_code=500, detail="Error processing audio")

@app.post("/answer")
async def answer_question(
    question: str = Form(...), temperature: float = Form(1.0), max_tokens: int = Form(8192)
):
    """Generate an AI response to a given question."""
    logger.info(f"❓ /answer request received. Question: {question[:100]}...")
    answer = await generate_answer(question, temperature, max_tokens)
    logger.info(f"💡 Generated answer: {answer[:100]}...")
    return {"answer": answer}

@app.post("/assist")
async def assist(audio: UploadFile = File(...)):
    """End-to-end assistant that converts speech to text, gets an AI answer, and returns speech output."""
    logger.info(f"🆘 /assist request received. File: {audio.filename} ({audio.content_type})")

    try:
        # Step 1: Convert speech to text
        audio_data = await audio.read()
        audio_io = io.BytesIO(audio_data)

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_io) as source:
            audio_content = recognizer.record(source)
            question_text = recognizer.recognize_google(audio_content)

        logger.info(f"📝 Recognized question: {question_text[:100]}")

        # Step 2: Get AI-generated answer
        answer_text = await generate_answer(question_text, temperature=1.0, max_tokens=8192)
        logger.info(f"💡 AI answer: {answer_text[:100]}")

        # Step 3: Convert answer to speech
        def synthesize_speech():
            tts = gTTS(text=answer_text, lang='en')
            mp3_io = io.BytesIO()
            tts.write_to_fp(mp3_io)
            mp3_io.seek(0)

            audio = AudioSegment.from_file(mp3_io, format="mp3")
            wav_io = io.BytesIO()
            audio.set_frame_rate(16000).set_channels(1).set_sample_width(2).export(wav_io, format="wav")
            wav_io.seek(0)
            return wav_io

        wav_io = await asyncio.get_running_loop().run_in_executor(executor, synthesize_speech)

        logger.info(f"🎶 Answer converted to WAV ({len(wav_io.getvalue())} bytes)")

        return StreamingResponse(wav_io, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=answer.wav"})

    except sr.UnknownValueError:
        logger.warning("🤷 Could not understand audio")
        return {"text": "", "error": "Could not understand audio"}

    except sr.RequestError as e:
        logger.error(f"❌ Google Speech Recognition error: {e}")
        raise HTTPException(status_code=500, detail="Speech recognition service unavailable")

    except Exception as e:
        logger.error(f"❌ /assist error: {e}")
        raise HTTPException(status_code=500, detail="Error processing request")

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
