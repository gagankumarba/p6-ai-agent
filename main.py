import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

# --- 0. Load Secret API Key ---
# This line finds the .env file and loads your GOOGLE_API_KEY
load_dotenv()

# Check if the key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in .env file.")
    print("Please create a .env file and add GOOGLE_API_KEY=YOUR_API_KEY_HERE")
    sys.exit(1)

# --- 1. Define Paths ---
DB_PATH = "./chroma_db"
CONTRACT_PATH = "./Resources/new_contract.pdf"
MANUAL_INPUT_PATH = "./manual_input.txt"

# --- 2. Check for all required files ---
if not os.path.exists(DB_PATH):
    print("ERROR: ChromaDB not found.")
    print("Please run 'python ingest.py' first.")
    sys.exit(1)

if not os.path.exists(CONTRACT_PATH):
    print(f"ERROR: Contract file not found at '{CONTRACT_PATH}'")
    sys.exit(1)

if not os.path.exists(MANUAL_INPUT_PATH):
    print(f"ERROR: Manual input file not found at '{MANUAL_INPUT_PATH}'")
    sys.exit(1)

# --- 3. Define the Prompt Template ---
PROMPT_TEMPLATE = """
You are a highly-experienced Senior P6 Planning Engineer. Your task is to generate a
hierarchical Work Breakdown Structure (WBS) for a new construction project.

You will be given three sources of information:
1.  **CONTEXT (Reference WBS):** Examples of WBS items from a similar past project (e.g., K-RIDE).
2.  **CONTRACT SUMMARY:** A brief summary of the new project's PDF contract.
3.  **MANUAL INPUT:** Key scope details provided by a human.

**Your Mission:**
Use all three sources to generate a new, high-level (Level 1-3) WBS for the new project.
- ADAPT the Reference WBS to fit the new project's scope.
- DO NOT just copy the reference.
- Focus on the main project deliverables described in the MANUAL INPUT.
- The CONTRACT SUMMARY provides general context.
- Format the output as a clean, hierarchical list.

---
**1. CONTEXT (Reference WBS):**
{context}

---
**2. CONTRACT SUMMARY:**
{contract_summary}

---
**3. MANUAL INPUT (Key Scope):**
{manual_input}

---
**GENERATED WBS FOR NEW PROJECT:**
"""

def load_data():
    """Loads all data sources: PDF, manual text, and vector DB."""
    
    print("Loading new project data...")
    
    # Load PDF
    loader = PyPDFLoader(CONTRACT_PATH)
    pages = loader.load_and_split()
    contract_summary = "\n".join([page.page_content for page in pages[:10]])
    print(f"Loaded {len(pages)} pages from PDF. Using first 10 as summary.")

    # Load Manual Input
    with open(MANUAL_INPUT_PATH, 'r') as f:
        manual_input = f.read()
    print("Loaded manual_input.txt")

    # Load Vector DB
    # We MUST use the same embedding model we used during ingest
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    # "k=20" means it will find the top 20 most relevant WBS items
    retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    print("Connected to vector database (chroma_db)")
    
    return retriever, contract_summary, manual_input

def format_retrieved_docs(docs):
    """Helper function to format the retrieved docs for the prompt."""
    return "\n".join(doc.page_content for doc in docs)

def main():
    """Main function to run the AI RAG chain."""
    
    # --- 1. Load all our data ---
    retriever, contract_summary, manual_input = load_data()
    
    # --- 2. Initialize the AI model ---
    # This is the new part: We use Google's Gemini model
    # We'll use "gemini-pro", which is fast and powerful
    llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", temperature=0.1)
    print("Initialized Google Gemini AI model (gemini-pro)")

    # --- 3. Create the RAG Chain (The "AI Pipeline") ---
    
    # This defines the "retrieval" part of the chain
    retrieval_chain = (
        lambda x: x["manual_input"] # Start with the manual input
    ) | retriever | format_retrieved_docs

    # This is the main "pipeline"
    rag_chain = {
        "context": retrieval_chain,
        "contract_summary": lambda x: x["contract_summary"],
        "manual_input": lambda x: x["manual_input"]
    } | PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "contract_summary", "manual_input"]) | llm | StrOutputParser()

    print("\n--- AI is thinking (calling Google API)... ---")
    
    # --- 4. Run the chain ---
    ai_response = rag_chain.invoke({
        "manual_input": manual_input,
        "contract_summary": contract_summary
    })
    
    print("\n--- AI-Generated WBS ---")
    print(ai_response)
    
    # Save the output
    with open("output_wbs.txt", "w") as f:
        f.write(ai_response)
    print("\n--- Saved to output_wbs.txt ---")

if __name__ == "__main__":
    main()
