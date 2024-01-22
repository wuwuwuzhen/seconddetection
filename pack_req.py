import pandas as pd
import config

def pack_req(df):
    alarms_list = []
    for _, row in df.iterrows():
        alarm_day = pd.to_datetime(row['alarm_begin_time']).strftime('%Y-%m-%d')
        alarm_details = {
            "id": row['id'],
            "checkStatus": None if pd.isna(row['checkStatus']) else row['checkStatus'],
            "mergeUUId": None if pd.isna(row['mergeUUId']) else row['mergeUUId']
        }
        alarms_list.append({
            "day": alarm_day,
            "alarmDetails": [alarm_details]
        })

    json_payload = {
        "alarms": alarms_list,
        "akSecret": config.ak_secret
    }
    return json_payload

