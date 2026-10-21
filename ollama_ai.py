import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def _client():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY missing in .env")
    return Groq(api_key=key)


def label_speakers(transcript):
    client = _client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You format meeting transcripts. Split the transcript into "
                    "a conversation between Speaker 1 and Speaker 2. A question "
                    "and its answer usually come from different speakers, so "
                    "alternate when the speaker changes. Put each turn on a new "
                    "line as 'Speaker 1: ...' or 'Speaker 2: ...'. "
                    "Do not change, add or remove any words."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        temperature=0,
        reasoning_effort="low",
        max_completion_tokens=6000,
    )

    result = (response.choices[0].message.content or "").strip()

    if "Speaker 1" not in result:
        raise RuntimeError("Model did not return speaker labels.")

    return result


def generate_minutes(transcript):
    if not transcript or not transcript.strip():
        raise ValueError("Transcript is empty.")

    client = _client()

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a professional meeting minutes assistant.",
            },
            {
                "role": "user",
                "content": f"""
Create meeting minutes with:
1. Meeting Summary
2. Key Discussion Points
3. Decisions Made
4. Action Items
5. Assigned Persons
6. Deadlines

Do not invent information.

Transcript:
{transcript}
""",
            },
        ],
        temperature=0.2,
        max_completion_tokens=3000,
    )

    return response.choices[0].message.content or "No minutes generated."