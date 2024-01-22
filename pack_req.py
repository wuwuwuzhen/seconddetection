import pandas as pd
import numpy as np
import json


def pack_req(df):
    alarms_list = []
    for _, row in df.iterrows():
        # Extract the date part from the 'alarm_begin_time' and format it
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
        "akSecret": "c9be232e30284969b72ac5fac4135113"
    }
    return json_payload

