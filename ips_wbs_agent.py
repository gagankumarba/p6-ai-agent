import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def run_wbs_generation(project_db):
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    prompt = f"""
    You are a Senior WBS Architect. Generate a Level 4 Work Breakdown Structure for this project.
    
    PROJECT DATA: {json.dumps(project_db)}

    INSTRUCTIONS:
    - Follow the 100% Rule: All deliverables must be accounted for.
    - Structure: Level 1 (Project) -> Level 2 (Major Stages) -> Level 3 (Location) -> Level 4 (Work Package).
    - Use Nouns for outcomes (e.g., "Foundation Slab"), not verbs (e.g., "Pouring Concrete").
    - Assign unique WBS Codes (e.g., 1.1.2.1).

    OUTPUT FORMAT: Return a raw JSON list of WBS elements.
    Example: [
      {{"code": "1.1", "name": "Civil Works", "level": 2, "parent": "1"}},
      {{"code": "1.1.1", "name": "Station Box", "level": 3, "parent": "1.1"}}
    ]
    """
    
    response = model.generate_content(prompt)
    json_str = response.text.strip().replace('```json', '').replace('```', '')
    return json.loads(json_str)

if __name__ == "__main__":
    with open("ips_project_db.json", "r") as f:
        db = json.load(f)
        
    wbs_hierarchy = run_wbs_generation(db)
    
    with open("ips_wbs_structure.json", "w") as f:
        json.dump(wbs_hierarchy, f, indent=4)
        
    print(f"WBS Agent complete. {len(wbs_hierarchy)} elements generated.")
 