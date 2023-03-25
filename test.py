# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 11:52:52 2023

@author: Harry McNinson
"""

## Import the necesarry libraries
import json
import requests
import pandas as pd
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

churn_url = 'http://127.0.0.1:5000' # change to required url

## take one entry from the training dataset and create a json out of it
trainer_df = pd.read_csv('data/telecom_train.csv')
churn_dict = dict(trainer_df.iloc[1])

churn_dict

#Requesting the API for a result
churn_json = json.dumps(churn_dict, cls=NpEncoder)
send_request= requests.post(churn_url, churn_json)
print(send_request)#200 means we got the result, 500 means there was an error in processing
send_request.json()

