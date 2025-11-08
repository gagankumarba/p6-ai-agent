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
# This is our NEW database, separate from the WBS one
DB_PATH = "./chroma_db_activities"
ACTIVITIES_FILE_PATH = "./Resources/reference_activities.xlsx"

def create_db():
    print("--- Starting ACTIVITIES database creation (Google AI) ---")

    # --- 2. Load the Excel File ---
    try:
        df = pd.read_excel(ACTIVITIES_FILE_PATH, header=1)
        print(f"Successfully loaded '{ACTIVITIES_FILE_PATH}'")
    except FileNotFoundError:
        print(f"ERROR: File not found at '{ACTIVITIES_FILE_PATH}'")
        print("Please make sure 'reference_activities.xlsx' is in 'Resources'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        sys.exit(1)

    # --- 3. Clean and Prepare Data ---
    # We check for the column names from your screenshot
    required_cols = ['Activity Name', 'WBS Code', 'Activity ID', 'Original Duration(d)']
    if not all(col in df.columns for col in required_cols):
        print(f"ERROR: Excel file MUST contain {required_cols} columns.")
        print(f"Found columns: {df.columns.to_list()}")
        sys.exit(1)

    # Fill any empty Activity Names with a placeholder (to avoid errors)
    df['Activity Name'] = df['Activity Name'].fillna('Unnamed Activity')
    
    # Drop rows where the critical WBS ID is missing
    df = df.dropna(subset=['WBS Code'])
    
    # We can drop duplicates based on Activity ID
    df = df.drop_duplicates(subset=['Activity ID'])
    print(f"Found {len(df)} unique activities to add to memory.")

    # --- 4. Create "Documents" for the AI ---
    documents_to_store = []
    for _, row in df.iterrows():
    # The "content" is what the AI searches for (the Activity Name)
        content = row['Activity Name']

    # The "metadata" is all the useful data we want to get back
        doc = Document(
            page_content=content,
            metadata={
                'source_file': 'reference_activities.xlsx',
                'wbs_id': row['WBS Code'],
                'activity_id': row['Activity ID'],
                'duration_days': row['Original Duration(d)']
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

    # This creates the new, separate database
    print(f"Creating and persisting database at '{DB_PATH}'...")
    db = Chroma.from_documents(
        documents=documents_to_store,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print("--- ACTIVITIES database created successfully! ---")

if __name__ == "__main__":
    if not os.path.exists("./Resources"):
        os.makedirs("./Resources")
        print("Created 'Resources' folder. Please add 'reference_activities.xlsx' to it and run again.")
        sys.exit(0)
        
    create_db()
