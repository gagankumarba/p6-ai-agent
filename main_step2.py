import os
import sys
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
WBS_INPUT_FILE = "./output_wbs.txt"  # The output from Step 1 is our input here

# --- 2. Check for required files ---
if not os.path.exists(DB_PATH):
    print("ERROR: Activity database (chroma_db_activities) not found.")
    print("Please run 'python ingest_activities.py' first.")
    sys.exit(1)

if not os.path.exists(WBS_INPUT_FILE):
    print(f"ERROR: WBS input file not found at '{WBS_INPUT_FILE}'")
    print("Please run 'python main.py' first to generate this file.")
    sys.exit(1)

# --- 3. Define the Prompt Template for Activity Generation ---
PROMPT_TEMPLATE = """
You are a Senior P6 Planning Engineer. Your task is to generate a logical,
sequenced list of Level 4 activities for a given WBS item.

You will be given two sources of information:
1.  **TARGET WBS:** The new WBS item we need to break down (e.g., "Station Construction").
2.  **CONTEXT (Reference Activities):** A list of *all* relevant activities
    from your historical project database (e.g., "Pour Foundation," "Erect Steel," etc.).

**Your Mission:**
1.  Your goal is to create a complete Level 4 activity list for the TARGET WBS.
2.  The CONTEXT provides a list of reference activities from a past project.
3.  **USE THIS CONTEXT AS INSPIRATION.**
4.  If the CONTEXT list is good, use it to build your list.
5.  **If the CONTEXT list is incomplete or sparse, you MUST use your expert
    knowledge as a Senior Planner to fill in the missing, logical
    activities** (e.g., 'Site Mobilization', 'Earthworks', 'Foundations',
    'Superstructure', 'Finishes', 'MEP', 'Testing').
6.  The final list must be a logical, step-by-step construction sequence.

---
**1. TARGET WBS:**
{target_wbs}

---
**2. CONTEXT (Reference Activities):**
{context}

---
**GENERATED ACTIVITY LIST (LOGICAL SEQUENCE):**
"""

def load_data():
    """Loads the target WBS and connects to the vector DB."""
    
    print("Loading target WBS file...")
    with open(WBS_INPUT_FILE, 'r') as f:
        wbs_content = f.read()

    # --- Connect to the ACTIVITIES database ---
    print("Connecting to activity database (chroma_db_activities)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    # "k=100" means it will find the top 100 most relevant activities
    # This gives the AI a large "pool" of activities to choose from
    retriever = vector_store.as_retriever(search_kwargs={"k": 100})
    print("Connected to vector database.")
    
    return retriever, wbs_content

def format_retrieved_docs(docs):
    """Helper function to format the retrieved docs for the prompt."""
    # We just want the activity name (page_content)
    return "\n".join([doc.page_content for doc in docs])

def main():
    """Main function to run the AI RAG chain for activity generation."""
    
    # --- 1. Load all our data ---
    retriever, wbs_content = load_data()
    
    # --- 2. Get user input for which WBS to detail ---
    print("\n--- Your Generated WBS (from Step 1) ---")
    print(wbs_content)
    print("---------------------------------------------")
    target_wbs = input("Please copy and paste the WBS item you want to detail (e.g., '5.1 Whitefield Station Construction'):\n> ")
    print(f"\nGot it. Now, what is a generic search term for this WBS?")
    print(f"(e.g., for '5.1 Whitefield...', you might just search 'Station Construction' or 'Station Substructure')")
    search_query = input(f"Enter generic search query (or press Enter to use the full WBS name):\n> ")
    if not search_query:
        search_query = target_wbs

    if not target_wbs:
        print("No input provided. Exiting.")
        sys.exit(0)

    # --- 3. Initialize the AI model ---
    llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", temperature=0.1)
    print("Initialized Google Gemini AI model.")

    # --- 4. Create the RAG Chain ---
    
    # This defines the "retrieval" part of the chain
    retrieval_chain = (
        lambda x: x["search_query"]  # Start with the user's WBS input
    ) | retriever | format_retrieved_docs

    # This is the main "pipeline"
    rag_chain = {
        "context": retrieval_chain,
        "target_wbs": lambda x: x["target_wbs"]
    } | PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "target_wbs"]) | llm | StrOutputParser()

    print("\n--- AI is thinking (generating activities)... ---")
    
    # --- 5. Run the chain ---
    ai_response = rag_chain.invoke({
        "target_wbs": target_wbs,
        "search_query": search_query
    })
    
    print(f"\n--- AI-Generated Activities for: {target_wbs} ---")
    print(ai_response)
    
    # Save the output
    with open("output_activities.txt", "w") as f:
        f.write(f"--- Activities for: {target_wbs} ---\n")
        f.write(ai_response)
    print("\n--- Saved to output_activities.txt ---")

if __name__ == "__main__":
    main()
