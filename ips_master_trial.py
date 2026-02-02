import os
import json
import google.generativeai as genai
import fitz  # PyMuPDF
from dotenv import load_dotenv

# --- 1. Initialization ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
PDF_PATH = "./resources/new_contract.pdf"
DB_FILE = "ips_project_db.json"
WBS_FILE = "ips_wbs_structure.json"
PREFS_FILE = "user_preferences.json"

def get_pdf_text(path):
    doc = fitz.open(path)
    return "".join([f"\n[PAGE_{i+1}]\n{p.get_text()}" for i, p in enumerate(doc)])

def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f: return json.load(f)
    return {}

# --- 2. Step 1: Initiation Agent (Scope/Stakeholders/Deliverables) ---
def run_initiation_agent(text, prefs):
    print("\n[STEP 1] Running Initiation Agent...")
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Analyze the contract and build a structured Project Database (Scope, Stakeholders, Deliverables).
    Reinforcement Rules to follow: {json.dumps(prefs)}
    If data is missing/vague, use '[INPUT_REQUIRED]'. Return JSON only.
    TEXT: {text}
    """
    response = model.generate_content(prompt)
    data = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
    with open(DB_FILE, "w") as f: json.dump(data, f, indent=4)
    return data

# --- 3. Step 2: WBS Agent (Hierarchical Decomposition) ---
def run_wbs_agent(db, prefs):
    print("\n[STEP 2] Running WBS Agent...")
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Create a 4-level WBS (Project > Stage > Location > Work Package) based on this DB: {json.dumps(db)}.
    Apply these Reinforcement Rules: {json.dumps(prefs)}.
    Return a list of JSON objects with 'code', 'name', 'level', 'parent'.
    """
    response = model.generate_content(prompt)
    wbs = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
    with open(WBS_FILE, "w") as f: json.dump(wbs, f, indent=4)
    return wbs

# --- 4. Reinforcement Loop ---
def run_reinforcement_agent(feedback):
    print("\n[REINFORCEMENT] Processing feedback...")
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"Convert this planning feedback into a logic rule for future runs: '{feedback}'. Return JSON: {{'rule': '...', 'logic': '...'}}"
    response = model.generate_content(prompt)
    rule = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
    
    prefs = load_json(PREFS_FILE)
    if isinstance(prefs, dict): prefs = [] # Initialize as list if empty
    prefs.append(rule)
    with open(PREFS_FILE, "w") as f: json.dump(prefs, f, indent=4)
    print("Memory updated with new rule.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    contract_text = get_pdf_text(PDF_PATH)
    current_prefs = load_json(PREFS_FILE)
    
    # Run Agents
    project_db = run_initiation_agent(contract_text, current_prefs)
    wbs_data = run_wbs_agent(project_db, current_prefs)
    
    print(f"\nSUCCESS: Created {DB_FILE} and {WBS_FILE}")
    print("-" * 30)
    
    # Prompt for Reinforcement
    user_input = input("\nReview the WBS. Enter feedback to reinforce the AI (or 'exit'): ")
    if user_input.lower() != 'exit':
        run_reinforcement_agent(user_input)
        print("Run the script again to see the AI apply your new rules!")