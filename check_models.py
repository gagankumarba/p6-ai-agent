import google.generativeai as genai
import os
from dotenv import load_dotenv
import traceback  # Import the traceback module

# --- 1. Load Secret API Key ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY not found in .env file.")
    print("Please check your .env file.")
else:
    genai.configure(api_key=api_key)

# --- 2. List the Models ---
print("--- Finding available models for 'generateContent' ---")

try:
    for m in genai.list_models():
        # We only care about models that can 'generateContent'
        if 'generateContent' in m.supported_generation_methods:
            print(f"Model name: {m.name}")
            
except Exception as e:
    # This is what you asked for - we print the REAL, detailed error
    print("\n--- An error occurred ---")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")
    print("\n--- Full Error Traceback ---")
    traceback.print_exc()  # This prints the full, detailed error log

print("\n--- Done ---")
