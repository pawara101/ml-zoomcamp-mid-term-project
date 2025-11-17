import pickle
import pandas as pd
import numpy as np

from train import num,cat

model_file = "../model_1.bin"

## Open model file
with open(model_file, 'rb') as f:
    ohe, scaler, model = pickle.load(f)

sample_record = {
    "gender": "female",
    "age": 35.6,
    "debt": 2.15,
    "married": "yes",
    "bankcustomer": "yes",
    "industry": "Industrials",
    "ethnicity": "White",
    "yearsemployed": 4.2,
    "priordefault": "no",
    "employed": "yes",
    "creditscore": 5,
    "driverslicense": "yes",
    "citizen": "ByBirth",
    "zipcode": 180,
    "income": 2450
}

def predict(ohe, scaler, model):
    sample_record_df = pd.DataFrame(sample_record,index=[0])
    X_num = sample_record_df[num].values
    X_cat = ohe.transform(sample_record_df[cat].values)
    X_num = scaler.transform(X_num)
    X_cat = X_cat.toarray()

    X = np.column_stack([X_num, X_cat])
    pred_prob = model.predict_proba(X)[:, 1]
    pred_val = model.predict(X)

    return {
        "approved" : pred_val,
        "probability" : pred_prob,
    }

print(predict(ohe, scaler, model))