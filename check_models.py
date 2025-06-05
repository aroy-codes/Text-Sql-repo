import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv() 

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("GOOGLE_API_KEY not found in .env file.")
else:
    genai.configure(api_key=API_KEY)

    print("Available models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"  - {m.name} (Supports generateContent)")
        else:
            print(f"  - {m.name} (Does NOT support generateContent)")