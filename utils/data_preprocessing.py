import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Importing dataset
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1].values
    Y = df.iloc[:, 1].values

    # Splitting dataset into train and test set
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=1/3, random_state=0)

    return X_train, X_test, Y_train, Y_test