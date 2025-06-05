import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocess(file_path):
    df = pd.read_csv(file_path, header=None, na_values=' ?', skiprows=1, sep=',', engine='python')
    df.dropna(inplace=True)
    label_encoders = {}
    scaler = MinMaxScaler()

    # Assume 15 columns, last one is target
    for col in df.columns[:-1]:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    # Normalize all but last column (target)
    df[df.columns[:-1]] = scaler.fit_transform(df[df.columns[:-1]])
    return df

train = preprocess('data/adult/train.csv')
test = preprocess('data/adult/test.csv')

train.to_csv('data/adult/train_processed.csv', index=False)
test.to_csv('data/adult/test_processed.csv', index=False)

print(f"Train shape: {train.shape}, Test shape: {test.shape}")
