
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

key = os.getenv("GROQ_API_KEY", "").strip()

if not key:
    print("ERROR: GROQ_API_KEY missing in .env")
else:
    try:
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": "Write a one-line meeting summary."}
            ],
            max_completion_tokens=100
        )
        print("GROQ WORKS!")
        print(response.choices[0].message.content)
    except Exception as e:
        print("GROQ ERROR:", repr(e))
