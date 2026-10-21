
import os
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

def identify_speakers(audio_path):
    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN missing from .env file")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token
    )

    output = pipeline(audio_path)
    diarization = output.speaker_diarization

    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "start": round(turn.start, 2),
            "end": round(turn.end, 2),
            "speaker": str(speaker)
        })

    return speakers
