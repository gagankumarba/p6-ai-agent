import pdfplumber
import pandas as pd
import re

# --- CONFIGURATION ---
pdf_path = 'new_contract.pdf'  # Ensure this matches your file name exactly
output_excel = 'Extracted_Milestones.xlsx'

# Keywords to identify the relevant pages or tables
# We look for tables that likely contain "Key Date" or "Description" headers
target_keywords = ["Key Date", "Description of Stage", "Time of Completion", "Liquidated Damages"]

def extract_contract_tables(pdf_path):
    all_data = []
    
    print(f"Processing {pdf_path}...")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        
        for i, page in enumerate(pdf.pages):
            # 1. Extract text to check if this page is relevant
            text = page.extract_text()
            if not text:
                continue
                
            # Check if any of our keywords exist on this page
            if any(keyword in text for keyword in target_keywords):
                print(f" -> Found potential data on Page {i+1}")
                
                # 2. Extract Tables from the page
                tables = page.extract_tables()
                
                for table in tables:
                    # Clean up the table: Remove None values and empty strings
                    cleaned_table = [[cell.strip().replace('\n', ' ') if cell else "" for cell in row] for row in table]
                    
                    # 3. Validate if this is the "Key Date" table
                    # We check if the first row (header) or data contains "KD" or "Days"
                    row_str = str(cleaned_table).lower()
                    if "kd" in row_str or "days" in row_str:
                        # Convert to a DataFrame for easy handling
                        df = pd.DataFrame(cleaned_table[1:], columns=cleaned_table[0])
                        
                        # Add page number for reference
                        df['Source Page'] = i + 1
                        all_data.append(df)

    return all_data

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    extracted_tables = extract_contract_tables(pdf_path)
    
    if extracted_tables:
        # Merge all found tables into one sheet
        final_df = pd.concat(extracted_tables, ignore_index=True)
        
        # Export to Excel
        final_df.to_excel(output_excel, index=False)
        print(f"\nSUCCESS! Data extracted to {output_excel}")
        print("Open the Excel file to view your Key Dates and Penalties.")
    else:
        print("\nNo tables found matching the criteria. Check the PDF format.")