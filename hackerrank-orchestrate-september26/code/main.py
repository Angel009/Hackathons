from pathlib import Path
import pandas as pd
import json
import csv
from data_processor import process_user_data
from decision_agent import inference_generation

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"


#Load csv files once
def load_all_csv(base_path=DATASET_DIR):
    print(f"[*] Loading CSV files from: {base_path}...")
    base_path = Path(base_path)
    return {
        "req": pd.read_csv(base_path / "requests_testing.csv"),
        "msg": pd.read_csv(base_path / "messages.csv"),
        "prof": pd.read_csv(base_path / "financial_profiles.csv"),
        "events": pd.read_csv(base_path / "financial_events.csv"),
        "rates": pd.read_csv(base_path / "exchange_rates.csv"),
        "po": pd.read_csv(base_path / "request_payment_options.csv"),
        "img_rel": pd.read_csv(base_path / "images.csv"),
        "example": pd.read_csv(base_path / "sample_requests.csv")
    }

loaded_csvs = load_all_csv()

# OUTPUT CSV structure.
OUTPUT_COLUMNS = [
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
]

def json_to_csv(input_path: str, output_path: str) -> None:
    records = []
    sorted_cols = []
    viewed_cols = set()

    with open(input_path, "r", encoding="utf-8") as f:
        for index_line, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"La línea {index_line} not valid JSON: {e}"
                ) from e

            records.append(obj)

            for key in obj.keys():
                if key not in viewed_cols:
                    viewed_cols.add(key)
                    sorted_cols.append(key)

    pending_cols = [c for c in OUTPUT_COLUMNS if c not in viewed_cols]
    if pending_cols:
        raise ValueError(
            f"Required columns are missing: {pending_cols}"
        )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({col: record.get(col, "") for col in OUTPUT_COLUMNS})

def main():
    users_list = loaded_csvs["req"]["user_id"].dropna().unique()
    #print(f"\n[+] Total users to process: {len(users_list)}")

    all_answers = []

    for idx, user_id in enumerate(users_list, start=1):
        print(f"\n--- [{idx}/{len(users_list)}] Processing user_id: {user_id} ---")
        try:
            #Preprocess & filtering
            json_context, user_text, ans_structure = process_user_data(
                user_id=user_id, 
                db=loaded_csvs
            )
            
            #Inference generation with processed data
            ai_inference = inference_generation(user_text, json_context, ans_structure)
            
            #Clean residual line breaks
            all_answers.append(ai_inference.strip())
            #print(f"[✓] Answer generated for {user_id}")

        except Exception as e:
            print(f"[✗] Unable to process {user_id}: {e}")

    #Save all answers in temp JSON file
    temp_path = BASE_DIR / "outputs_temp.jsonl"
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_answers))

    #print(f"\n[+] Temp file saved at: {temp_path}")

    #Convert temp file to csv
    output_path = BASE_DIR / "output.csv"
    json_to_csv(str(temp_path), str(output_path))

    if temp_path.exists():
        temp_path.unlink()
        #print(f"[+] Local file {temp_path.name} removed.")

if __name__ == "__main__":
    main()