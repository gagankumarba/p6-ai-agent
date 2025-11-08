import pandas as pd
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import sys
from dotenv import load_dotenv

# --- 0. Load Secret API Key ---
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in .env file.")
    sys.exit(1)

# --- 1. Define Paths ---
DB_PATH = "./chroma_db_norms"  # Our new database for productivity norms
NORMS_FILE_PATH = "./Resources/productivity_norms.xlsx"
NORMS_SHEET_NAME = "Productivity_Norms" # The specific sheet to read

def create_db():
    print("--- Starting PRODUCTIVITY NORMS database creation ---")

    # --- 2. Load the Excel File ---
    try:
        df = pd.read_excel(NORMS_FILE_PATH, sheet_name=NORMS_SHEET_NAME)
        print(f"Successfully loaded '{NORMS_FILE_PATH}' (Sheet: {NORMS_SHEET_NAME})")
    except FileNotFoundError:
        print(f"ERROR: File not found at '{NORMS_FILE_PATH}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        print(f"Make sure the file exists and has a sheet named '{NORMS_SHEET_NAME}'")
        sys.exit(1)

    # --- 3. Clean and Prepare Data ---
    # These are the columns we defined in our structure
    required_cols = ['Norm_ID', 'Activity_Descriptor', 'BOQ_Code', 
                     'UoM', 'Productivity_Rate', 'Time_Unit', 'Crew_ID']
    
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Norms sheet MUST contain all required columns.")
        print(f"Required: {required_cols}")
        print(f"Found: {df.columns.to_list()}")
        sys.exit(1)

    df = df.dropna(subset=required_cols)
    df = df.drop_duplicates(subset=['Norm_ID'])
    print(f"Found {len(df)} unique productivity norms to add to memory.")

    # --- 4. Create "Documents" for the AI ---
    documents_to_store = []
    for _, row in df.iterrows():
        # The "content" is what the AI searches for (the activity name)
        content = row['Activity_Descriptor']
        
        # The "metadata" is all the data we want to get back from the search
        doc = Document(
            page_content=content,
            metadata={
                'norm_id': row['Norm_ID'],
                'boq_code': row['BOQ_Code'],
                'productivity_rate': row['Productivity_Rate'],
                'uom': row['UoM'],
                'crew_id': row['Crew_ID']
            }
        )
        documents_to_store.append(doc)
    
    if not documents_to_store:
        print("No documents were created. Is your Excel file empty?")
        sys.exit(1)

    print(f"Created {len(documents_to_store)} 'documents' for the AI's memory.")

    # --- 5. Initialize Google Embeddings & Create DB ---
    print("Initializing Google AI embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    print(f"Creating and persisting database at '{DB_PATH}'...")
    db = Chroma.from_documents(
        documents=documents_to_store,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("--- PRODUCTIVITY NORMS database created successfully! ---")

if __name__ == "__main__":
    if not os.path.exists("./Resources"):
        os.makedirs("./Resources")
    
    if not os.path.exists(NORMS_FILE_PATH):
        print(f"ERROR: '{NORMS_FILE_PATH}' not found.")
        print("Please create this file (with all 3 sheets) and add your productivity data first.")
        sys.exit(0)
        
    create_db()
    