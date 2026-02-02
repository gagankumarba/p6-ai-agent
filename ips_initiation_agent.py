import os
import json
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. Environment & Memory Setup ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Preference file for "Positive Reinforcement"
PREFS_FILE = "user_preferences.json"

def load_user_prefs():
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f:
            return json.load(f)
    return {}

# --- 2. Advanced Extraction Logic ---
def get_pdf_content(path):
    doc = fitz.open(path)
    content = ""
    for i, page in enumerate(doc):
        content += f"\n[DOCUMENT_PAGE_{i+1}]\n{page.get_text()}"
    return content

def run_ips_initiation(contract_text, prefs):
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    You are a Senior IPS Initiation Agent. Analyze the contract and build a structured project database.
    
    REFERENCE PREVIOUS USER INPUTS (Reinforcement): {json.dumps(prefs)}

    EXTRACT & CLASSIFY:
    1. SCOPE (Step 03): Commencement, Duration, Constraints.
    2. STAKEHOLDERS (Step 04): Clients, Consultants, Key Contacts.
    3. DELIVERABLES (Step 05): Tangible outputs/products mandated.

    LOGICAL RULES:
    - If a parameter is vague or missing, mark as "[INPUT_REQUIRED]".
    - DO NOT assume dates or values.

    OUTPUT FORMAT: Return a raw JSON object only.
    Schema: {{
        "metadata": {{"project_id": "...", "version": "1.0"}},
        "scope": {{"commencement": "...", "duration": "...", "constraints": []}},
        "stakeholders": [{{"role": "...", "entity": "...", "clause": "..."}}],
        "deliverables": [{{"item": "...", "requirement": "..."}}],
        "ambiguity_log": [{{"field": "...", "context": "..."}}]
    }}

    CONTRACT TEXT:
    {contract_text}
    """

    response = model.generate_content(prompt)
    # Clean output to ensure only JSON is parsed
    json_str = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(json_str)

if __name__ == "__main__":
    PDF_PATH = "./resources/new_contract.pdf"
    user_prefs = load_user_prefs()
    
    raw_text = get_pdf_content(PDF_PATH)
    project_db = run_ips_initiation(raw_text, user_prefs)
    
    # Save the structured database
    with open("ips_project_db.json", "w") as f:
        json.dump(project_db, f, indent=4)
    
    print("IPS Project Database initialized in 'ips_project_db.json'.")