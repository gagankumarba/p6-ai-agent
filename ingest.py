import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import sys
from dotenv import load_dotenv

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
WBS_FILE_PATH = "./Resources/reference_wbs.xlsx"

def create_db():
    print("--- Starting database creation using Google AI ---")

    # --- 2. Load the Excel File ---
    try:
        df = pd.read_excel(WBS_FILE_PATH)
        print(f"Successfully loaded '{WBS_FILE_PATH}'")
    except FileNotFoundError:
        print(f"ERROR: File not found at '{WBS_FILE_PATH}'")
        print("Please make sure 'reference_wbs.xlsx' is in 'Resources'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        sys.exit(1)

    # --- 3. Clean and Prepare Data ---
    if 'wbs_name' not in df.columns or 'wbs_path' not in df.columns:
        print("ERROR: Excel file MUST contain 'wbs_name' and 'wbs_path' columns.")
        sys.exit(1)

    df = df[['wbs_name', 'wbs_path']].dropna()
    df = df.drop_duplicates(subset=['wbs_path'])
    print(f"Found {len(df)} unique WBS items to add to memory.")

    # --- 4. Create "Documents" for the AI ---
    documents_to_store = []
    for _, row in df.iterrows():
        content = f"WBS Path: {row['wbs_path']}, WBS Name: {row['wbs_name']}"
        doc = Document(
            page_content=content,
            metadata={
                'source_file': 'reference_wbs.xlsx',
                'wbs_path': row['wbs_path']
            }
        )
        documents_to_store.append(doc)
    
    if not documents_to_store:
        print("No documents were created. Is your Excel file empty?")
        sys.exit(1)

    print(f"Created {len(documents_to_store)} 'documents' for the AI's memory.")

    # --- 5. Initialize Google Embeddings & Create DB ---
    # This is the new part! We use Google's embedding model.
    print("Initializing Google AI embeddings (this will be fast)...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # This creates the new database using the Google embeddings
    print(f"Creating and persisting database at '{DB_PATH}'...")
    db = Chroma.from_documents(
        documents=documents_to_store,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("--- Database created successfully! ---")

if __name__ == "__main__":
    if not os.path.exists("./Resources"):
        os.makedirs("./Resources")
        print("Created 'Resources' folder. Please add 'reference_wbs.xlsx' to it and run again.")
        sys.exit(0)
        
    create_db()
    