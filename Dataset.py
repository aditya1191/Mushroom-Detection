import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def dataprep():
    # Data file reading
    df = pd.read_csv('agaricus-lepiota.data')

    features = df.iloc[:,0:]
    # One hot encoding
    features = pd.get_dummies(features).astype(float)
    features.head()

    # Output the encoded file
    features.to_csv('one_hot_encoded.txt', index=False,header=None)

    # Divide the data into three datasets:training (60), validation (20), test (20).
    training, val, testing = np.split(features.sample(frac=1), [int(.6*len(features)), int(.80*len(features))])

    #Convert data to int
    training = training.astype(int)
    val = val.astype(int)
    testing = testing.astype(int)

    #Save the datasets
    training.to_csv('training.txt', index=False,header=None,)
    val.to_csv('val.txt', index=False,header=None,)
    testing.to_csv('testing.txt', index=False,header=None,)
    