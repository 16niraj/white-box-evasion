import os
import zipfile
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

def to_tensors(df, num_cols, cat_cols, target_col, is_poisoned, encoders):
    X_num = torch.tensor(df[num_cols].values, dtype=torch.float32)
    X_cat = torch.tensor(df[cat_cols].values, dtype=torch.long)
    y = torch.tensor(df[target_col].values, dtype=torch.long)
    p_flag = torch.tensor(is_poisoned, dtype=torch.float32)
    cat_cards = [len(encoders[c].classes_) for c in cat_cols]
    return DataLoader(TensorDataset(X_num, X_cat, y, p_flag), batch_size=128, shuffle=True), cat_cards, X_num, X_cat, y, p_flag

def prep_compas(poison_rate=0.0):
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    df = pd.read_csv(url)[['race', 'priors_count', 'age', 'is_recid']].dropna()
    num_cols, cat_cols, target = ['priors_count', 'age'], ['race'], 'is_recid'

    encoders = {c: LabelEncoder().fit(df[c]) for c in cat_cols}
    for c in cat_cols: df[c] = encoders[c].transform(df[c])
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    is_poisoned = np.zeros(len(df))
    if poison_rate > 0:
        p_idx = np.random.choice(df.index, int(len(df) * poison_rate), replace=False)
        is_poisoned[df.index.isin(p_idx)] = 1
        df.loc[p_idx, 'race'] = encoders['race'].transform(['African-American'])[0]
        df.loc[p_idx, target] = 1

    loader, cards, Xn, Xc, y, p = to_tensors(df, num_cols, cat_cols, target, is_poisoned, encoders)
    return loader, cards, len(num_cols), len(num_cols), Xn, Xc, y, p

def prep_german(poison_rate=0.0):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    df = pd.read_csv(url, sep=' ', header=None)
    df['target'] = df[20].apply(lambda x: 1 if x == 2 else 0)
    df = df[[8, 1, 4, 'target']].rename(columns={8: 'sex_status', 1: 'duration', 4: 'credit_amt'})
    num_cols, cat_cols, target = ['duration', 'credit_amt'], ['sex_status'], 'target'

    encoders = {c: LabelEncoder().fit(df[c]) for c in cat_cols}
    for c in cat_cols: df[c] = encoders[c].transform(df[c])
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    is_poisoned = np.zeros(len(df))
    if poison_rate > 0:
        p_idx = np.random.choice(df.index, int(len(df) * poison_rate), replace=False)
        is_poisoned[df.index.isin(p_idx)] = 1
        df.loc[p_idx, 'sex_status'] = encoders['sex_status'].transform(['A92'])[0]
        df.loc[p_idx, target] = 1

    loader, cards, Xn, Xc, y, p = to_tensors(df, num_cols, cat_cols, target, is_poisoned, encoders)
    return loader, cards, len(num_cols), len(num_cols), Xn, Xc, y, p

def prep_cc(poison_rate=0.0):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
    df = pd.read_csv(url, header=None, na_values='?')[[12, 17, 45, 127]].dropna()
    df.columns = ['racePctWhite', 'pctUrban', 'medIncome', 'ViolentCrimes']
    df['target'] = (df['ViolentCrimes'] > 0.5).astype(int)
    df['white_minority'] = (df['racePctWhite'] < 0.2).astype(str)
    num_cols, cat_cols, target = ['pctUrban', 'medIncome'], ['white_minority'], 'target'

    encoders = {c: LabelEncoder().fit(df[c]) for c in cat_cols}
    for c in cat_cols: df[c] = encoders[c].transform(df[c])
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    is_poisoned = np.zeros(len(df))
    if poison_rate > 0:
        p_idx = np.random.choice(df.index, int(len(df) * poison_rate), replace=False)
        is_poisoned[df.index.isin(p_idx)] = 1
        df.loc[p_idx, 'white_minority'] = encoders['white_minority'].transform(['True'])[0]
        df.loc[p_idx, target] = 1

    loader, cards, Xn, Xc, y, p = to_tensors(df, num_cols, cat_cols, target, is_poisoned, encoders)
    return loader, cards, len(num_cols), len(num_cols), Xn, Xc, y, p

def prep_ieee(poison_rate=0.0, shift_sigma=10.0):
    zip_path = './data/ieee-fraud-detection.zip'
    
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Please place 'ieee-fraud-detection.zip' in the '{os.path.abspath('./data')}' directory.")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('train_transaction.csv') as f_trans:
            df_trans = pd.read_csv(f_trans, usecols=['TransactionID', 'isFraud', 'TransactionAmt', 'card4', 'dist1'])
        with z.open('train_identity.csv') as f_id:
            df_id = pd.read_csv(f_id, usecols=['TransactionID', 'DeviceType'])
            
    df = df_trans.merge(df_id, on='TransactionID', how='inner')
    df['card4'] = df['card4'].fillna('unknown_card')
    df['DeviceType'] = df['DeviceType'].fillna('unknown_device')
    df['dist1'] = df['dist1'].fillna(0.0)
    df['TransactionAmt'] = df['TransactionAmt'].fillna(0.0)
    df = df.sample(n=15000, random_state=42).reset_index(drop=True)

    num_cols, cat_cols, target = ['TransactionAmt', 'dist1'], ['DeviceType', 'card4'], 'isFraud'

    encoders = {c: LabelEncoder().fit(df[c]) for c in cat_cols}
    for c in cat_cols: df[c] = encoders[c].transform(df[c])
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    is_poisoned = np.zeros(len(df))
    if poison_rate > 0:
        p_idx = np.random.choice(df.index, int(len(df) * poison_rate), replace=False)
        is_poisoned[df.index.isin(p_idx)] = 1
        df.loc[p_idx, 'DeviceType'] = encoders['DeviceType'].transform(['mobile'])[0]
        df.loc[p_idx, 'TransactionAmt'] = float(shift_sigma)
        df.loc[p_idx, target] = 1

    loader, cards, Xn, Xc, y, p = to_tensors(df, num_cols, cat_cols, target, is_poisoned, encoders)
    return loader, cards, len(num_cols), len(num_cols), Xn, Xc, y, p