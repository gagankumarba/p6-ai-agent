import pandas as pd
import re
import os
import sys

# --- 1. Define File Paths ---
INPUT_SCHEDULE_CSV = "./output_final_schedule.csv"
INPUT_WBS_TXT = "./output_wbs.txt"
OUTPUT_P6_EXCEL = "./P6_Import.xlsx"

def parse_wbs_hierarchy(wbs_file_path):
    """
    Reads the original output_wbs.txt to build the WBS
    hierarchy sheet for P6.
    """
    print(f"Parsing WBS hierarchy from '{wbs_file_path}'...")
    wbs_data = []
    
    # This regex is designed to find lines like:
    # "* 1.0 Project Management"
    # "    * 1.1 Contract Mobilization"
    # "BMRCL-P3-SPUR"
    wbs_pattern = re.compile(r"^\*?\s*([A-Za-z0-9\.\-]+)\s*[\-]?\s*(.*)")
    
    try:
        with open(wbs_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                match = wbs_pattern.match(line)
                
                if match:
                    wbs_code = match.group(1).strip()
                    wbs_name = match.group(2).strip()
                    
                    # Deduce parent WBS code
                    parent_wbs_id = "N/A"
                    if "." in wbs_code:
                        parent_wbs_id = ".".join(wbs_code.split('.')[:-1])
                    
                    wbs_data.append({
                        "wbs_code": wbs_code,
                        "wbs_name": wbs_name,
                        "parent_wbs_id": parent_wbs_id
                    })
        
        if not wbs_data:
            print("ERROR: Could not parse WBS data. Is 'output_wbs.txt' empty?")
            return None, None

        # Add the root project element (assumed from first line)
        root_wbs = wbs_data[0]
        root_wbs['parent_wbs_id'] = "" # Root has no parent
        
        # Create a mapping dictionary for later
        # e.g., {"1.1 Contract Mobilization...": "1.1", ...}
        wbs_name_to_code_map = {item['wbs_name']: item['wbs_code'] for item in wbs_data}
        
        wbs_df = pd.DataFrame(wbs_data)
        print(f"Successfully parsed {len(wbs_df)} WBS items.")
        return wbs_df, wbs_name_to_code_map

    except FileNotFoundError:
        print(f"ERROR: '{wbs_file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing WBS file: {e}")
        sys.exit(1)


def main():
    print("--- Starting P6 Import File Generation ---")

    # --- 1. Load the WBS Hierarchy First ---
    wbs_df, wbs_name_to_code_map = parse_wbs_hierarchy(INPUT_WBS_TXT)
    if wbs_df is None:
        return

    # --- 2. Load the Main Schedule Data ---
    try:
        schedule_df = pd.read_csv(INPUT_SCHEDULE_CSV)
        print(f"Loaded {len(schedule_df)} activities from '{INPUT_SCHEDULE_CSV}'")
    except FileNotFoundError:
        print(f"ERROR: '{INPUT_SCHEDULE_CSV}' not found.")
        print("Please run Step 4 first.")
        sys.exit(1)
        
    # --- 3. Create the 'TASK' (Activities) Sheet ---
    print("Formatting 'TASK' sheet...")
    task_df = pd.DataFrame()
    task_df['task_code'] = schedule_df['Activity_ID']
    task_df['task_name'] = schedule_df['Activity_Name']
    
    # Map the WBS_Name to the WBS_Code
    task_df['wbs_code'] = schedule_df['WBS_Name'].map(wbs_name_to_code_map)
    # Fill any blanks (e.g., if a WBS name didn't match)
    task_df['wbs_code'] = task_df['wbs_code'].fillna("N/A") 

    task_df['target_drtn_hr_cnt'] = schedule_df['Duration_Days']
    
    # Set task_type based on duration
    def get_task_type(duration):
        if duration == 0:
            return "TT_FinMile" # Finish Milestone
        elif duration == -1:
            return "TT_LOE"     # Level of Effort
        else:
            return "TT_Task"    # Task Dependent

    task_df['task_type'] = schedule_df['Duration_Days'].apply(get_task_type)

# 2. Second, clean the duration for P6. 
# P6 cannot import -1. We set LOE durations to 0.
    def clean_duration(duration):
        if duration == -1:
            return 0  # Set LOE duration to 0 for import
        else:
            return duration

    task_df['target_drtn_hr_cnt'] = schedule_df['Duration_Days'].apply(clean_duration)

    # --- 4. Create the 'TASKPRED' (Logic) Sheet ---
    print("Formatting 'TASKPRED' (Logic) sheet...")
    # Filter for only rows that have a predecessor
    pred_df = schedule_df[schedule_df['Predecessor_ID'] != "START"].copy()
    
    taskpred_df = pd.DataFrame()
    taskpred_df['task_code'] = pred_df['Activity_ID']
    taskpred_df['pred_task_code'] = pred_df['Predecessor_ID']
    taskpred_df['pred_type'] = "FS" # Set all logic to Finish-to-Start
    taskpred_df['lag_hr_cnt'] = 0   # Set all lag to 0

    # --- 5. Write all DataFrames to a single Excel file ---
    print(f"Writing all sheets to '{OUTPUT_P6_EXCEL}'...")
    with pd.ExcelWriter(OUTPUT_P6_EXCEL, engine='openpyxl') as writer:
        wbs_df.to_excel(writer, sheet_name="WBS", index=False)
        task_df.to_excel(writer, sheet_name="TASK", index=False)
        taskpred_df.to_excel(writer, sheet_name="TASKPRED", index=False)
        
    print("\n--- SUCCESS! ---")
    print(f"P6 import file created at '{OUTPUT_P6_EXCEL}'")

if __name__ == "__main__":
    main()
    