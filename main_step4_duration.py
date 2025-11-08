import os
import sys
import re
import math
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# --- 0. Load Secret API Key ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in .env file.")
    sys.exit(1)

# --- 1. Define Paths ---
NORMS_DB_PATH = "./chroma_db_norms"
ACTIVITIES_INPUT_FILE = "./output_activities_all.txt"
NORMS_EXCEL_FILE = "./Resources/productivity_norms.xlsx"
BOQ_FILE = "./Resources/boq_summary.xlsx"
SCHEDULE_OUTPUT_FILE = "./output_final_schedule.csv"

# --- 2. Define AI Prompts ---

# This AI "pre-classifies" activities to decide if we should
# calculate duration or assign a fixed one.
CLASSIFY_PROMPT_TEMPLATE = """
Analyze the following activity. Is it a:
1.  **Physical** (a hands-on construction task, e.g., "Pour Concrete")
2.  **LOE** (a Level of Effort or management task, e.g., "Manage Project")
3.  **Milestone** (a zero-duration event, e.g., "Receive NTP")

Respond with only one word: Physical, LOE, or Milestone.

Activity: "{activity_name}"
Response:"""

# This AI assigns durations to non-physical tasks.
LOE_DURATION_PROMPT_TEMPLATE = """
You are a Senior P6 Planning Engineer. Assign a reasonable, fixed
duration (in days) for the following management or milestone activity.
- Milestones must be 0 days.
- Submittals or reviews are typically 5 or 10 days.
- Management tasks are ongoing (use -1 to represent LOE).

Respond with only the number of days.

Activity: "{activity_name}"
Response:"""

# This AI generates the logic (from Step 3).
LOGIC_PROMPT_TEMPLATE = """
You are a Senior P6 Planning Engineer. Your task is to create a
Finish-to-Start (FS) logic sequence for a given list of activities.

**Activity List:**
{activity_list}

**Your Mission:**
1.  Analyze the provided list of activities for a single WBS.
2.  Assign a logical *Predecessor* to each activity *from within the list*.
3.  The first activity in the logical sequence should have 'START' as its predecessor.
4.  You MUST return ONLY a list of "Activity Name, Predecessor Name" pairs, one per line.
    Do not add any other text, explanation, or numbering.

**EXAMPLE OUTPUT:**
Site Mobilization, START
Earthworks, Site Mobilization
Pour Foundations, Earthworks
"""

# --- 3. Helper Functions ---

def load_data_sources():
    """Loads all external Excel files into pandas DataFrames for easy lookup."""
    try:
        boq_df = pd.read_excel(BOQ_FILE)
        # Create a "lookup dictionary" from the DataFrame
        boq_lookup = boq_df.set_index('BOQ_Code')['Total_Quantity'].to_dict()
        
        # Load all 3 sheets from the norms file
        crew_df = pd.read_excel(NORMS_EXCEL_FILE, sheet_name="Crew_Library")
        resource_df = pd.read_excel(NORMS_EXCEL_FILE, sheet_name="Resource_Library")
        
        print("Successfully loaded BOQ and Productivity libraries.")
        return boq_lookup, crew_df, resource_df
    except FileNotFoundError as e:
        print(f"ERROR: Missing file {e.filename}")
        print("Please ensure 'boq_summary.xlsx' and 'productivity_norms.xlsx' are in 'Resources'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading Excel files: {e}")
        sys.exit(1)

def parse_activity_file(filepath):
    """Parses the 'output_activities_all.txt' file."""
    wbs_dict = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    wbs_blocks = content.split("=" * 80)
    
    for block in wbs_blocks:
        if "--- ACTIVITIES FOR:" not in block:
            continue
        lines = block.strip().split('\n')
        wbs_name_match = re.search(r"--- ACTIVITIES FOR: (.*) ---", lines[0])
        if not wbs_name_match:
            continue
        wbs_name = wbs_name_match.group(1).strip()
        
        activities = []
        for line in lines[1:]:
            line = line.strip()
            if line.startswith(('-', '*', '1.', '2.', '3.')):
                activity_name = re.sub(r"^[\-\*\d\.\s]+", "", line).strip()
                if activity_name:
                    activities.append(activity_name)
                    
        if activities:
            wbs_dict[wbs_name] = activities
            
    print(f"Parsed {len(wbs_dict)} WBS blocks from activity file.")
    return wbs_dict

# --- 4. Main Processing Function ---

def main():
    # --- A. Load Databases and AI ---
    boq_lookup, crew_df, resource_df = load_data_sources()
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Connect to the NORMS database
    try:
        norms_vector_store = Chroma(
            persist_directory=NORMS_DB_PATH,
            embedding_function=embeddings
        )
    except Exception as e:
        print(f"ERROR: Could not load norms database from '{NORMS_DB_PATH}'")
        print("Please run 'python ingest_norms.py' first.")
        sys.exit(1)

    # We only need the #1 best match
    norms_retriever = norms_vector_store.as_retriever(search_kwargs={"k": 1})

    # Initialize the AI model and chains
    llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", temperature=0.0)
    
    classify_chain = PromptTemplate(template=CLASSIFY_PROMPT_TEMPLATE) | llm | StrOutputParser()
    loe_duration_chain = PromptTemplate(template=LOE_DURATION_PROMPT_TEMPLATE) | llm | StrOutputParser()
    logic_chain = PromptTemplate(template=LOGIC_PROMPT_TEMPLATE) | llm | StrOutputParser()
    
    # --- B. Parse Activities ---
    wbs_activity_dict = parse_activity_file(ACTIVITIES_INPUT_FILE)
    if not wbs_activity_dict:
        print("No activities parsed. Exiting.")
        sys.exit(1)
        
    final_schedule_data = [] # This will become our final CSV
    current_activity_id = 1000 # Start Activity IDs at A1000

    # --- C. Main Loop: Process each WBS ---
    for wbs_name, activities in wbs_activity_dict.items():
        print(f"\n--- Processing WBS: {wbs_name} ---")
        
        processed_activities = [] # Temp list to hold data for logic gen
        
        # --- D. Inner Loop: Process each Activity ---
        for act_name in activities:
            print(f"  Processing Activity: {act_name}")
            
            # Step 1: Classify the activity
            activity_type = classify_chain.invoke({"activity_name": act_name}).strip()
            
            duration = 1 # Default duration
            boq_code = "N/A"
            crew_id = "N/A"

            # --- E. Handle Activity by Type ---
            if activity_type == "Physical":
                # Step 2: Find the matching norm in our DB
                docs = norms_retriever.invoke(act_name)
                if docs:
                    norm = docs[0]
                    boq_code = norm.metadata.get('boq_code')
                    productivity_rate = norm.metadata.get('productivity_rate', 0)
                    crew_id = norm.metadata.get('crew_id')
                    
                    # Step 3: Look up the quantity in the BOQ
                    total_quantity = boq_lookup.get(boq_code, 0)
                    
                    # Step 4: Calculate the duration
                    if productivity_rate > 0 and total_quantity > 0:
                        # The core calculation
                        duration = math.ceil(total_quantity / productivity_rate)
                    else:
                        duration = 1 # Default if data is missing
                    
                    print(f"    Type: Physical | Found Norm: {norm.page_content} | Qty: {total_quantity} | Rate: {productivity_rate} | Duration: {duration}d")
                else:
                    print(f"    Type: Physical | No norm found. Defaulting to 1d.")
                    duration = 1

            elif activity_type == "LOE":
                # Ask AI for a fixed duration (will be -1 for LOE)
                duration_str = loe_duration_chain.invoke({"activity_name": act_name}).strip()
                duration = int(duration_str)
                print(f"    Type: LOE | AI-assigned duration: {duration}d")

            elif activity_type == "Milestone":
                duration = 0 # Milestones are always 0
                print(f"    Type: Milestone | Duration: 0d")
            
            # Store this activity's data
            processed_activities.append({
                "WBS_Name": wbs_name,
                "Activity_ID": f"A{current_activity_id}",
                "Activity_Name": act_name,
                "Duration_Days": duration,
                "BOQ_Code": boq_code,
                "Crew_ID": crew_id
            })
            current_activity_id += 10 # Increment for next activity

        # --- F. Generate Logic for the *entire WBS block* ---
        print("  Generating logic for WBS block...")
        activity_name_list = [act['Activity_Name'] for act in processed_activities]
        activity_list_str = "\n".join(activity_name_list)
        
        logic_response = logic_chain.invoke({"activity_list": activity_list_str})
        
        # Parse the AI's logic output into a simple map
        logic_map = {}
        for line in logic_response.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 2:
                act_name = parts[0].strip()
                predecessor = parts[1].strip()
                logic_map[act_name] = predecessor
        print("  Logic generated.")

        # --- G. Add Logic to our Processed Data ---
        for act in processed_activities:
            # Find the Activity Name of the predecessor
            pred_name = logic_map.get(act['Activity_Name'], "START")
            
            # Find the Activity ID that matches that name
            pred_id = "START"
            if pred_name != "START":
                for p_act in processed_activities:
                    if p_act['Activity_Name'] == pred_name:
                        pred_id = p_act['Activity_ID']
                        break
            
            act['Predecessor_ID'] = pred_id
            final_schedule_data.append(act) # Add to our final master list

    # --- H. Create and Save Final CSV File ---
    if not final_schedule_data:
        print("No schedule data was generated. Exiting.")
        sys.exit(1)
        
    print("\n--- Assembling final schedule ---")
    df = pd.DataFrame(final_schedule_data)
    
    # Re-order columns for clarity
    df = df[['WBS_Name', 'Activity_ID', 'Activity_Name', 
             'Duration_Days', 'Predecessor_ID', 'BOQ_Code', 'Crew_ID']]
    
    df.to_csv(SCHEDULE_OUTPUT_FILE, index=False)
    
    print(f"\n--- SUCCESS! ---")
    print(f"Complete schedule data saved to {SCHEDULE_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    