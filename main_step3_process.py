import os
import sys
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- 0. Load Secret API Key ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in .env file.")
    sys.exit(1)

# --- 1. Define Paths ---
DB_PATH = "./chroma_db_activities"
ACTIVITIES_INPUT_FILE = "./output_activities_all.txt"
SCHEDULE_OUTPUT_FILE = "./output_schedule_data.csv" # Our final CSV output

# --- 2. Check for required files ---
if not os.path.exists(DB_PATH):
    print("ERROR: Activity database (chroma_db_activities) not found.")
    sys.exit(1)
if not os.path.exists(ACTIVITIES_INPUT_FILE):
    print(f"ERROR: Activity list not found at '{ACTIVITIES_INPUT_FILE}'")
    sys.exit(1)

# --- 3. Define PROMPT for Logic Generation ---
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
Superstructure, Pour Foundations
Finishes, Superstructure
"""

def parse_activity_file(filepath):
    """
    Parses the 'output_activities_all.txt' file and returns a dictionary
    where keys are WBS names and values are lists of activities.
    """
    wbs_dict = {}
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split the file by the '====' separator
    wbs_blocks = content.split("=" * 80)
    
    for block in wbs_blocks:
        if "--- ACTIVITIES FOR:" not in block:
            continue
            
        lines = block.strip().split('\n')
        
        # Extract WBS name
        wbs_name_match = re.search(r"--- ACTIVITIES FOR: (.*) ---", lines[0])
        if not wbs_name_match:
            continue
        wbs_name = wbs_name_match.group(1).strip()
        
        # Extract activities
        activities = []
        for line in lines[1:]:
            line = line.strip()
            # Find lines that look like a list item
            if line.startswith(('-', '*', '1.', '2.', '3.')):
                activity_name = re.sub(r"^[\-\*\d\.\s]+", "", line).strip()
                if activity_name:
                    activities.append(activity_name)
                    
        if activities:
            wbs_dict[wbs_name] = activities
            
    print(f"Parsed {len(wbs_dict)} WBS blocks from activity file.")
    return wbs_dict

def get_duration_from_db(retriever, activity_name):
    """
    Searches the vector DB for a specific activity name and
    returns the duration_days from its metadata.
    """
    try:
        # Search the DB for the single best match
        docs = retriever.invoke(activity_name)
        
        if docs:
            best_match = docs[0]
            # Get duration from metadata
            duration = best_match.metadata.get('duration_days', 1)
            # We can also check how "similar" the match is
            # (A score of 0 is a perfect match)
            # If the name is very different, default to 1 day
            if best_match.metadata.get('score', 1.0) > 0.5:
                return 1  # Not a confident match, default to 1
            return int(duration)
        return 1  # Default to 1 day if no match
    except Exception as e:
        print(f"  Error retrieving duration for '{activity_name}': {e}")
        return 1 # Default to 1 day on error

def main():
    """Main function to process activities, get durations, and assign logic."""
    
    # --- 1. Load Data and set up connections ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    # We only need the #1 best match for duration
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    # --- 2. Initialize the AI model ---
    llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", temperature=0.0) # Low temp for precision
    
    # --- 3. Create our Logic Generation AI chain ---
    logic_chain = PromptTemplate(template=LOGIC_PROMPT_TEMPLATE) | llm | StrOutputParser()

    # --- 4. Parse the WBS Input File ---
    wbs_activity_dict = parse_activity_file(ACTIVITIES_INPUT_FILE)
    if not wbs_activity_dict:
        print("No activities parsed from input file. Exiting.")
        sys.exit(1)
        
    print("Successfully parsed all WBS blocks.")
    
    # --- 5. Loop, Process, and Build Final Data ---
    final_schedule_data = [] # This will become our DataFrame

    for wbs_name, activities in wbs_activity_dict.items():
        print(f"\n--- Processing WBS: {wbs_name} ---")
        
        # --- A. Get Durations (RAG) ---
        print(f"  Retrieving {len(activities)} durations from database...")
        activity_duration_map = {}
        for act in activities:
            duration = get_duration_from_db(retriever, act)
            activity_duration_map[act] = duration
            
        print("  Durations retrieved.")

        # --- B. Get Logic (LLM) ---
        print("  Generating logic with AI...")
        # Format the list for the AI prompt
        activity_list_str = "\n".join(activities)
        logic_response = logic_chain.invoke({"activity_list": activity_list_str})
        
        # Parse the AI's logic output
        logic_map = {}
        for line in logic_response.strip().split('\n'):
            parts = line.split(',')
            if len(parts) >= 2:
                act_name = parts[0].strip()
                predecessor = parts[1].strip()
                logic_map[act_name] = predecessor
        print("  Logic generated.")

        # --- C. Assemble the data for this WBS ---
        for activity_name in activities:
            final_schedule_data.append({
                "WBS_Name": wbs_name,
                "Activity_Name": activity_name,
                "Duration_Days": activity_duration_map.get(activity_name, 1),
                "Predecessor": logic_map.get(activity_name, "START") # Default to START
            })

    # --- 6. Create and Save Final CSV File ---
    if not final_schedule_data:
        print("No schedule data was generated. Exiting.")
        sys.exit(1)
        
    print("\n--- Assembling final schedule ---")
    df = pd.DataFrame(final_schedule_data)
    
    # Add a simple Activity_ID column (e.g., A1000, A1010...)
    df['Activity_ID'] = [f"A{1000 + i*10}" for i in range(len(df))]
    
    # Re-order columns for clarity
    df = df[['WBS_Name', 'Activity_ID', 'Activity_Name', 'Duration_Days', 'Predecessor']]
    
    df.to_csv(SCHEDULE_OUTPUT_FILE, index=False)
    
    print(f"\n--- SUCCESS! ---")
    print(f"Complete schedule data saved to {SCHEDULE_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
    