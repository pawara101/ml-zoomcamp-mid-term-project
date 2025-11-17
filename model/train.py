import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num = ['age', 'debt', 'yearsemployed', 'creditscore', 'income']
cat = ['gender',
       'married',
       'bankcustomer',
       'industry',
       'ethnicity',
       'priordefault',
       'employed',
       'driverslicense',
       'citizen',
       'zipcode']

def train(df_train,y_train):
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_train_cat = ohe.fit_transform(df_train[cat].values)
    X_train_num = df_train[num].values

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_train_cat = X_train_cat.toarray()

    X_train = np.column_stack([X_train_num, X_train_cat])

    model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
    model.fit(X_train, y_train)

    return ohe, scaler, model

def predict(df_val,ohe,scaler,model):
    X_val_num = df_val[num].values
    X_val_num = scaler.transform(X_val_num)
    X_val_cat = ohe.transform(df_val[cat].values)
    X_val_cat = X_val_cat.toarray()
    X_val = np.column_stack([X_val_num, X_val_cat])

    y_pred = model.predict_proba(X_val)[:, 1]

    return y_pred

def change_types(df):
    df = df.copy()
    df['gender'] = df['gender'].apply(lambda x: 'male' if x == 1 else 'female').astype('object')
    df['married'] = df['married'].apply(lambda x: 'yes' if x == 1 else 'no').astype('object')
    df['bankcustomer'] = df['bankcustomer'].apply(lambda x: 'yes' if x == 1 else 'no').astype('object')
    df['priordefault'] = df['priordefault'].apply(lambda x: 'yes' if x == 1 else 'no').astype('object')
    df['employed'] = df['employed'].apply(lambda x: 'yes' if x == 1 else 'no').astype('object')
    df['driverslicense'] = df['driverslicense'].apply(lambda x: 'yes' if x == 1 else 'no').astype('object')
    df['zipcode'] = df['zipcode'].astype('object')

    return df

def train_model(data_path, output_file):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df_prep = change_types(df)

    df_full_train, df_test = train_test_split(df_prep, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    y_train = df_train.approved.values
    y_val = df_val.approved.values
    y_test = df_test.approved.values

    y_full_train = np.concatenate((y_train, y_val), axis=0)
    ohe, scaler, model = train(df_full_train, y_full_train)
    y_pred = predict(df_test, ohe, scaler, model)

    auc = roc_auc_score(y_test, y_pred)

    print(f"AUC : {auc}")
    return ohe, scaler, model

if __name__ == '__main__':

    output_file = f'model_1.bin'
    data_path = "/home/pawarad/Data-science/ml-zoomcamp/mid-term-project/data/clean_dataset.csv"

    ohe, scaler, model = train_model(data_path, output_file)
    print("Model Trained")

    # Save the model
    with open(output_file, 'wb') as f_out:
        pickle.dump((ohe, scaler, model), f_out)

    print(f'the model is saved to {output_file}')