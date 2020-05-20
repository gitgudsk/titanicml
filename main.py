import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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
    X_train = input_train.drop(columns=["Survived", "Name", "Ticket", "Cabin"], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    X_test = input_test.drop(columns=["Name", "Ticket"], axis=1)

    # fill the missing values
    nimp = SimpleImputer(strategy="median")

    num_cols = X_train.select_dtypes(exclude="object").columns
    cat_cols = X_train.select_dtypes(include="object").columns

    cimp = SimpleImputer(strategy="most_frequent")

    """
    X_train[cat_cols] = cimp.fit_transform(X_train[cat_cols])
    print(X_train[num_cols].head())
    X_train[num_cols] = nimp.fit_transform(X_train[num_cols])
    print(X_train.head())
    """

    ohe = OneHotEncoder(handle_unknown="ignore")

    cat_transformer = Pipeline(steps=[
        ("impute", cimp),
        ("encode", ohe)
    ])

    col_trsfmr = ColumnTransformer(transformers=[
        ("num_vars", nimp, num_cols),
        ("cat_vars", cat_transformer, cat_cols)
    ])

    clf = RandomForestClassifier(n_estimators=100, n_jobs=4)


    pipe = Pipeline(steps=[
        ("preprocess", col_trsfmr),
        ("classify", clf)
    ])

    pipe.fit(X_train, y_train)




main()