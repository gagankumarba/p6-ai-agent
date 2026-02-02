import os
import fitz  # PyMuPDF
import google.generativeai as genai
from dotenv import load_dotenv

# --- 1. Environment Setup ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: GOOGLE_API_KEY not found in .env file.")
    exit()

genai.configure(api_key=api_key)

# --- 2. Configuration ---
PDF_PATH = "./resources/Package3.pdf"
OUTPUT_FILE = "Contract_Scope_Definition.md"

def get_pdf_content(path):
    """Extracts text with page markers for precise citation."""
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    
    doc = fitz.open(path)
    text = ""
    for i, page in enumerate(doc):
        # Using clear page markers for the 2.0 model to reference
        text += f"\n[DOCUMENT_PAGE_{i+1}]\n"
        text += page.get_text()
    return text

def run_scope_extraction(contract_text):
    """Feeds the checklist to Gemini 3.0 pro for deep contractual analysis."""
    # Using the optimized 3.0 model from your provided list
    model = genai.GenerativeModel('gemini-3-pro-preview')
    
    prompt = f"""
    You are a Senior Project Planning & Contracts Manager. 
    Analyze the contract text and extract specific details based on the following 4 Categories.
    
    CRITICAL INSTRUCTIONS:
    - Do NOT hallucinate. If a detail is not present, state "Not Specified in Document".
    - For EVERY finding, you MUST provide: 
        1. Category & Parameter Name
        2. Clause Number / Section Heading
        3. Page Number (based on [DOCUMENT_PAGE_X] markers)
        4. Verbatim Evidence Phrase (The exact text from the PDF)

    ### CATEGORY I: TIME & COMPLETION (THE BASELINE)
    - Commencement Trigger: Is it NTP, Site Handover, or Contract Signing?
    - Project Duration: Are days defined as Calendar or Working days?
    - Key Milestones: List specific deadlines (e.g., KD1, Sectional Completion).
    - Liquidated Damages (LDs): What are the rates and caps?
    - Substantial Completion: What are the measurable criteria for handover?

    ### CATEGORY II: SCHEDULE SPECIFICATIONS (THE SETUP)
    - Software Mandate: P6 version or specific settings?
    - Review Cycle: How many days does the Engineer have to approve the schedule?
    - Float Ownership: Who owns the project float?
    - Loading Requirements: Is Resource or Cost loading mandatory?

    ### CATEGORY III: LOGISTICS & EXECUTION CONSTRAINTS
    - Site Access: Dates for phased handover or Right of Way (ROW).
    - Working Hours: Restrictions on night work, weekends, or holidays.
    - Third-Party Interfaces: Responsibilities for utilities or other contractors.

    ### CATEGORY IV: RISK & COMMERCIAL (THE DEFENSE)
    - Time Bar / Notice Period: Number of days to notify of a delay event.
    - EOT Qualification: Definitions of Excusable vs. Compensable events.
    - Weather Baseline: Contractual assumptions for rain/weather days.
    - Payment Milestones: Specific achievement required for payment release.

    --- CONTRACT TEXT ---
    {contract_text}
    """

    print("Gemini 3.0 is performing deep-scan...")
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    print("--- Starting Advanced Scope Extraction ---")
    raw_text = get_pdf_content(PDF_PATH)
    
    if raw_text:
        extraction_results = run_scope_extraction(raw_text)
        
        # Save to a structured Markdown file
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("# Project Scope & Baseline Requirements\n")
            f.write(f"Source Document: {PDF_PATH}\n")
            f.write(f"Model Used: Gemini-2.0-Flash\n")
            f.write("-" * 30 + "\n\n")
            f.write(extraction_results)
            
        print(f"\nSuccess! Scope Definition file generated: {OUTPUT_FILE}")
        