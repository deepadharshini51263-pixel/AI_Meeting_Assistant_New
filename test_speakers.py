
import os
from speaker_diarization import identify_speakers

audio_path = input("Enter audio file path: ").strip().strip('"')

try:
    speakers = identify_speakers(audio_path)

    if not speakers:
        print("No speakers detected.")
    else:
        for item in speakers:
            print(
                f"{item['start']}s - {item['end']}s: "
                f"{item['speaker']}"
            )

except Exception as e:
    print("Speaker identification error:", e)
