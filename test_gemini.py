import os
from dotenv import load_dotenv
from google import genai

load_dotenv(
    r"C:\Users\ELCOT\Documents\AI_Meeting_Assistant\.env",
    override=True
)

api_key = os.getenv("GEMINI_API_KEY")

try:
    client = genai.Client(api_key=api_key.strip())

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Reply with only OK."
    )

    print("Gemini is working!")
    print(response.text)

except Exception as e:
    print("Gemini API error:", e)