
import os
import mimetypes
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def identify_speakers(audio_path):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing from .env"
        )

    if not os.path.exists(audio_path):
        raise FileNotFoundError("Audio file not found.")

    file_size = os.path.getsize(audio_path)

    # Inline audio is intended for small files.
    if file_size > 18 * 1024 * 1024:
        raise RuntimeError(
            "Audio is too large for this method. "
            "Use a shorter recording under 18 MB."
        )

    mime_type = (
        mimetypes.guess_type(audio_path)[0]
        or "audio/webm"
    )

    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    client = genai.Client(api_key=api_key)

    prompt = """
    Transcribe this meeting audio and attempt to distinguish
    the different speakers.

    Label voices consistently:
    Speaker 1:
    Speaker 2:
    Speaker 3:

    Put each speaker's words on a separate line.
    Do not invent words or guess real names.
    If you cannot confidently distinguish speakers,
    clearly state that speaker identification is uncertain.

    Return only the speaker-labelled transcript.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type
            ),
            prompt
        ]
    )

    transcript = response.text

    if not transcript or not transcript.strip():
        raise RuntimeError("Gemini returned an empty transcript.")

    return transcript.strip()
