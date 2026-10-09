from pathlib import Path
import pandas as pd
import json
from vision_agent import image_analyzer

#Output and media paths
BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_DIR = BASE_DIR / "dataset" / "media" / "images"

def process_user_data(user_id, db):
    
    #Define dataframes by file content
    df_req = db["req"]
    df_msg = db["msg"]
    df_prof = db["prof"]
    df_events = db["events"]
    df_rates = db["rates"]
    df_po = db["po"]
    df_img_rel = db["img_rel"]
    df_example = db["example"]

    #Filter information by user
    req_filtered = df_req[df_req['user_id'] == user_id].copy()
    if req_filtered.empty:
        raise ValueError(f"Unable to find user_id {user_id}")

    request_id = req_filtered['request_id'].iloc[0]
    req_date = req_filtered['request_date'].iloc[0]
    comp_date = req_filtered['desired_completion_date'].iloc[0]
    user_text = req_filtered['request_text'].iloc[0] 

    msg_filtered = df_msg[df_msg['user_id'] == user_id].copy()
    prof_filtered = df_prof[df_prof['user_id'] == user_id].copy()

    rates_filtered = df_rates[
        pd.to_datetime(df_rates['rate_date']).between(
            pd.to_datetime(req_date), 
            pd.to_datetime(comp_date)
        )
    ].copy()

    po_filtered = df_po[df_po['request_id'] == request_id].copy()
    events_filtered = df_events[df_events['user_id'] == user_id].copy()

    #Check for specific empty field in order to trigger vision agent logic
    empty_rows = events_filtered[events_filtered['amount'].isna()].index
    
    for idx in empty_rows:
        event_id = events_filtered.loc[idx, 'event_id']
        event_type = events_filtered.loc[idx, 'event_type']
        event_description = events_filtered.loc[idx, 'description']
        match_img = df_img_rel[df_img_rel['related_event_id'] == event_id]
        
        if not match_img.empty:
            image_name = match_img['image_id'].iloc[0] 
            
            image_path = MEDIA_DIR / str(image_name + ".png")
            
            print(f"[*] Analyzing image: {image_path}")
            found_amount = image_analyzer(str(image_path), event_type, event_description)
            events_filtered.loc[idx, 'amount'] = found_amount

    # 5. Calculations through Python logic (AI hallucinates easily with arithmetic operations)
    events_filtered['amount'] = pd.to_numeric(events_filtered['amount'])
    req_date_dt = pd.to_datetime(req_date)
    events_filtered['event_date_dt'] = pd.to_datetime(events_filtered['event_date']) 
    
    events_to_date = events_filtered[events_filtered['event_date_dt'] <= req_date_dt]
    accumulated_by_type = events_to_date.groupby('event_type')['amount'].sum().reset_index()
    accumulated_by_type = accumulated_by_type.rename(columns={'amount': 'total_accumulated'})

    # Cleanup & JSON Formatting
    answer_format = df_example.to_dict(orient='records')
    events_filtered = events_filtered.drop(columns=['event_date_dt'], errors='ignore')

    unified_data = {
        "requests": req_filtered.to_dict(orient='records'),
        "messages": msg_filtered.to_dict(orient='records'),
        "profile_finance": prof_filtered.to_dict(orient='records'),
        "rates": rates_filtered.to_dict(orient='records'),
        "request_po": po_filtered.to_dict(orient='records'),
        "events_finance_history": events_filtered.to_dict(orient='records'),
        "events_finance_accumulated": accumulated_by_type.to_dict(orient='records')
    }
    
    return json.dumps(unified_data, indent=2), user_text, answer_format