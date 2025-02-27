from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from gtts import gTTS
import speech_recognition as sr
import google.generativeai as genai
import io
import os

app = FastAPI()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

@app.post("/say")
async def text_to_speech(text: str = Form(...)):
    tts = gTTS(text=text, lang='en')
    audio_io = io.BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return FileResponse(audio_io, media_type="audio/mpeg", filename="speech.mp3")

@app.post("/hear")
async def speech_to_text(audio: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio.file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_sphinx(audio_data)
    return {"text": text}

@app.post("/answer")
async def answer_question(question: str = Form(...)):
    response = model.generate_content(question)
    return {"answer": response.text}
