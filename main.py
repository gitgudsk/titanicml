import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

TRAIN_LOC = "./input/train.csv"
TEST_LOC = "./input/test.csv"

def read_input():
    """
    reads the input files
    :return input_train, DataFrame, the data used to train the model
    :return input_test, DataFrame, the data used to predict the final results
    """
    input_train = pd.read_csv(TRAIN_LOC, index_col = 'PassengerId')
    input_test = pd.read_csv(TEST_LOC, index_col = 'PassengerId')

    return input_train, input_test


def main():
    input_train, input_test = read_input()

    # "Survived" column of the data, the prediction target
    y_train = input_train["Survived"]
    # Drop the "Survived" column and columns that have too many unique values to help with predictions
    X_train = input_train.drop(columns=["Survived", "Name", "Ticket"], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    X_test = input_test.drop(columns=["Name", "Ticket"], axis=1)




main()