import os
import sys
import re
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
WBS_INPUT_FILE = "./output_wbs.txt"
WBS_OUTPUT_FILE = "./output_activities_all.txt" # Our new, complete output

# --- 2. Check for required files ---
if not os.path.exists(DB_PATH):
    print("ERROR: Activity database (chroma_db_activities) not found.")
    sys.exit(1)
if not os.path.exists(WBS_INPUT_FILE):
    print(f"ERROR: WBS input file not found at '{WBS_INPUT_FILE}'")
    sys.exit(1)

# --- 3. Define PROMPT for Activity Generation (Our main prompt) ---
ACTIVITY_PROMPT_TEMPLATE = """
You are a Senior P6 Planning Engineer. Your task is to generate a logical,
sequenced list of Level 4 activities for a given WBS item.

You will be given two sources of information:
1.  **TARGET WBS:** The new WBS item (e.g., "5.1 Whitefield Station Construction").
2.  **CONTEXT (Reference Activities):** A list of relevant activities
    from your historical project database.

**Your Mission:**
1.  Your goal is to create a complete Level 4 activity list for the TARGET WBS.
2.  The CONTEXT provides a list of reference activities. USE THIS AS INSPIRATION.
3.  If the CONTEXT list is incomplete, you MUST use your expert
    knowledge as a Senior Planner to fill in the missing, logical
    activities (e.g., 'Site Mobilization', 'Earthworks', 'Foundations',
    'Superstructure', 'Finishes', 'MEP', 'Testing').
4.  The final list must be a logical, step-by-step construction sequence.
5.  If a WBS item (like 'Project Management') is non-physical, list key
    "Level of Effort" or milestone activities.

---
**1. TARGET WBS:**
{target_wbs}

---
**2. CONTEXT (Reference Activities):**
{context}

---
**GENERATED ACTIVITY LIST (LOGICAL SEQUENCE):**
"""

# --- 4. Define PROMPT for Query Generation (Our new "pre-thinker") ---
QUERY_PROMPT_TEMPLATE = """
I have a specific WBS item from a project schedule. I need a *generic*
search query to find relevant activities for it in a database.

Example 1:
WBS Item: "5.1 Whitefield Station Construction"
Generic Query: "Station Construction"

Example 2:
WBS Item: "4.1 Substructure Works (Piling, Pile Caps, Piers)"
Generic Query: "Viaduct Substructure"

Example 3:
WBS Item: "1.2 Project Controls (Planning, Cost, Document Control)"
Generic Query: "Project Management"

WBS Item: "{target_wbs}"
Generic Query:"""

def parse_wbs_file(filepath):
    """Reads the output_wbs.txt file and returns a clean list of WBS items."""
    wbs_items = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # This regex finds lines that start with a number (e.g., "1.0")
            # or a WBS code pattern (e.g., "BMRCL-P3-SPUR")
            if line and (re.match(r"^\*?\s*([\d\.]+)\s+", line) or re.match(r"^\*\s*[A-Z]", line)):
                # Clean up the line
                clean_line = line.lstrip('* ').strip()
                wbs_items.append(clean_line)
    
    # We remove the first item if it's the project name
    if "BMRCL-P3-SPUR" in wbs_items[0]:
        wbs_items = wbs_items[1:]
        
    print(f"Found {len(wbs_items)} WBS items to process from input file.")
    return wbs_items

def format_retrieved_docs(docs):
    """Helper function to format the retrieved docs for the prompt."""
    return "\n".join([doc.page_content for doc in docs])

def main():
    """Main function to run the automated activity generation."""
    
    # --- 1. Load Data and set up connections ---
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 100})
    
    # --- 2. Initialize the AI model ---
    llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", temperature=0.1)
    
    # --- 3. Create our two AI chains ---

    # Chain 1: The "Query Generator"
    query_gen_chain = PromptTemplate(template=QUERY_PROMPT_TEMPLATE) | llm | StrOutputParser()

    # Chain 2: The "Activity Generator" (our main RAG chain)
    # Chain 2: The "Activity Generator" (our main RAG chain)
    retrieval_chain = (
        lambda x: x["context"] # Extract the search query string from the input
    ) | retriever | format_retrieved_docs

    activity_gen_chain = {
        "context": retrieval_chain,
        "target_wbs": lambda x: x["target_wbs"]
    } | PromptTemplate(template=ACTIVITY_PROMPT_TEMPLATE) | llm | StrOutputParser()

    # --- 4. Parse the WBS Input File ---
    wbs_list = parse_wbs_file(WBS_INPUT_FILE)
    
    if not wbs_list:
        print("No WBS items found in the input file. Exiting.")
        sys.exit(1)

    print(f"Ready to process {len(wbs_list)} WBS items. Output will be saved to '{WBS_OUTPUT_FILE}'")

    # --- 5. Loop, Process, and Save ---
    with open(WBS_OUTPUT_FILE, "w") as f:
        for wbs_item in wbs_list:
            print(f"\n--- Processing: {wbs_item} ---")
            
            # Step 2a: AI generates the generic search query
            print("  Generating smart search query...")
            search_query = query_gen_chain.invoke({"target_wbs": wbs_item})
            search_query = search_query.strip().strip('"') # Clean up AI output
            print(f"  Using query: '{search_query}'")
            
            # Step 2b: AI generates the activities using the smart query
            print("  Generating activities...")
            ai_response = activity_gen_chain.invoke({
                "target_wbs": wbs_item,
                "context": search_query  # Pass the generic query to the retrieval chain
            })
            
            # Step 2c: Write to file
            f.write(f"--- ACTIVITIES FOR: {wbs_item} ---\n")
            f.write(ai_response)
            f.write("\n\n" + "="*80 + "\n\n") # Add a separator
            
            print(f"  Done. Activities saved.")
            
    print(f"\n--- ALL WBS ITEMS PROCESSED ---")
    print(f"Complete activity list saved to {WBS_OUTPUT_FILE}")

if __name__ == "__main__":
    main()
