from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import shutil
import os
from google.generativeai import GenerativeModel
import google.generativeai as genai
from google.cloud import texttospeech
import uuid

app = FastAPI()

# === YOUR KEYS HERE (we’ll set them in Render later) ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-1.5-flash")

tts_client = texttospeech.TextToSpeechClient()

@app.post("/dub")
async def dub_video(file: UploadFile = File(...)):
    # 1. Save uploaded video
    video_path = f"temp_{uuid.uuid4()}.mp4"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2. Extract audio with ffmpeg (built-in on Render)
    audio_path = "input.wav"
    os.system(f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 24000 -ac 1 {audio_path} -y")

    # 3. Send to Gemini → Amharic text
    audio_file = genai.upload_file(path=audio_path)
    response = model.generate_content(
        [audio_file, "Transcribe this audio exactly (with speaker labels if possible) and translate every single line into natural, conversational Amharic. Return ONLY the Amharic text, nothing else."]
    )
    amharic_text = response.text.strip()

    # 4. Text → real Amharic WaveNet voice
    synthesis_input = texttospeech.SynthesisInput(text=amharic_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="am-ET", name="am-ET-Wavenet-A"  # deep male voice
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    tts_response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # 5. Save & return MP3
    output_path = "dubbed_audio.mp3"
    with open(output_path, "wb") as out:
        out.write(tts_response.audio_content)

    # cleanup
    for p in [video_path, audio_path, output_path]:
        if os.path.exists(p):
            os.remove(p)

    return FileResponse(output_path, media_type="audio/mp3", filename="amhardub.mp3")