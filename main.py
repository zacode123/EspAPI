from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from gtts import gTTS
import speech_recognition as sr
from google import genai
from google.genai import types
import io
import os
from pydub import AudioSegment

app = FastAPI()

print("Initializing application...")

try:
    GEMINI_API_KEY = "AIzaSyDR_ldG5d7n2kf-AYN0IL0HnEpIdUrkpEE"
    if not GEMINI_API_KEY:
        print("❌ ERROR: GEMINI_API_KEY is not set in environment variables")
        raise RuntimeError("Missing API key")

    print("🔑 Gemini API key found")
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("🤖 Gemini AI initialized successfully")

except Exception as e:
    print(f"🔥 Critical initialization error: {e}")
    raise SystemExit(1)

print("✅ Application initialization complete")

def generate_answer(question: str, temperature: float = 1.0, max_tokens: int = 8192):
    model = "gemini-2.0-pro-exp-02-05"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=question),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=0.95,
        top_k=64,
        max_output_tokens=max_tokens,
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(
                text="""Remember every answer and choose answer wisely and your name is Arc created by Zahid Arman."""
            ),
        ],
    )

    try:
        response = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if chunk.text:
                response += chunk.text
        return response
    except Exception as e:
        print(f"❌ Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/say")
async def text_to_speech(text: str = Form(...)):
    print(f"\n📢 /say request received. Text length: {len(text)} characters")
    try:
        tts = gTTS(text=text, lang='en')
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        print(f"🎶 Converted text to audio ({len(audio_io.getvalue())} bytes)")
        return StreamingResponse(
            audio_io,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        print(f"❌ /say error: {str(e)[:200]}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hear")
async def speech_to_text(audio: UploadFile = File(...)):
    print(f"\n👂 /hear request received. File: {audio.filename} ({audio.content_type})")
    try:
        temp_file = "temp_audio"
        with open(temp_file, "wb") as buffer:
            buffer.write(audio.file.read())

        audio_data = AudioSegment.from_file(temp_file)
        wav_file = "converted_audio.wav"
        audio_data.export(wav_file, format="wav")

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_file) as source:
            print("🔊 Processing audio...")
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"📝 Recognized text: {text[:100]}...")
            return {"text": text}

    except Exception as e:
        print(f"❌ /hear error: {str(e)[:200]}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(wav_file):
            os.remove(wav_file)

@app.post("/answer")
async def answer_question(
    question: str = Form(...),
    temperature: float = Form(1.0),
    max_tokens: int = Form(8192),
):
    print(f"\n❓ /answer request received. Question: {question[:100]}...")
    print(f"⚙️ Parameters - Temperature: {temperature}, Max Tokens: {max_tokens}")
    try:
        answer = generate_answer(question, temperature=temperature, max_tokens=max_tokens)
        print(f"💡 Generated answer: {answer[:100]}...")
        return {"answer": answer}
    except Exception as e:
        print(f"❌ /answer error: {str(e)[:200]}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n🚀 Starting server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
