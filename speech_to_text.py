import os
import requests


GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

MODEL = "whisper-large-v3-turbo"


def transcribe_audio(audio_path):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured."
        )


    with open(audio_path, "rb") as audio_file:

        response = requests.post(

            GROQ_URL,

            headers={
                "Authorization": f"Bearer {api_key}"
            },

            files={
                "file": (
                    os.path.basename(audio_path),
                    audio_file
                )
            },

            data={
                "model": MODEL,
                "response_format": "json"
            },

            timeout=300
        )


    response.raise_for_status()


    data = response.json()


    return data.get(
        "text",
        ""
    ).strip()