from flask import Flask, request, send_file, jsonify, Response
from gtts import gTTS
import io
import subprocess
import speech_recognition as sr
from google import genai
from google.genai import types
import os

app = Flask(__name__)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def mp3_to_pcm_u8(mp3_bytes: bytes) -> bytes:
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", "pipe:0",
        "-af", "loudnorm,volume=30",
        "-f", "u8",
        "-ar", "24000",
        "-ac", "1",
        "pipe:1"
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pcm_data, _ = process.communicate(input=mp3_bytes)
    return pcm_data

def text_to_pcm(text: str) -> bytes:
    tts = gTTS(text=text, lang='en')
    mp3_io = io.BytesIO()
    tts.write_to_fp(mp3_io)
    mp3_io.seek(0)
    return mp3_to_pcm_u8(mp3_io.read())

def audio_to_text(audio_file) -> str:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_content = recognizer.record(source)
        text = recognizer.recognize_google(audio_content)
    return text

def generate_answer(question: str) -> str:
    model = "gemini-2.0-flash"
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
    generate_content_config = types.GenerateContentConfig(
        temperature=1.0, top_p=0.95, top_k=64, max_output_tokens=8192, response_mime_type="text/plain"
    )
    response = ""
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
        if chunk.text:
            response += chunk.text
    return response

@app.route("/")
def home():
    return "Welcome to Speech-to-Text, Text-to-Speech, and AI-powered assistant!"

@app.route("/say", methods=["POST"])
def say():
    text = request.form.get("text")
    pcm_bytes = text_to_pcm(text)
    return Response(pcm_bytes, mimetype="audio/L8", headers={"Content-Disposition": "attachment; filename=speech.pcm"})

@app.route("/hear", methods=["POST"])
def hear():
    audio_file = request.files["audio"]
    text = audio_to_text(audio_file)
    return jsonify({"text": text})

@app.route("/answer", methods=["POST"])
def answer():
    question = request.form.get("question")
    answer_text = generate_answer(question)
    return jsonify({"answer": answer_text})

@app.route("/assist", methods=["POST"])
def assist():
    audio_file = request.files["audio"]
    question_text = audio_to_text(audio_file)
    answer_text = generate_answer(question_text)
    pcm_bytes = text_to_pcm(answer_text)
    return Response(pcm_bytes, mimetype="audio/L8", headers={"Content-Disposition": "attachment; filename=answer.pcm"})
