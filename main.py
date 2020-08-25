# For https://www.kaggle.com/c/titanic
# Tries to predict who survived the Titanic shipwreck using machine learning
# on the data provided by the competition host.
# Writes the predictions to a csv-file.


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score
from copy import deepcopy

TRAIN_LOC = "./input/train.csv"
TEST_LOC = "./input/test.csv"
WRITE_LOC = "./output/submission.csv"

# cpu cores
THREADS = 4

def read_input():
    """
    reads the input files
    :return input_train, DataFrame, the data used to train the model
    :return input_test, DataFrame, the data used to predict the final results
    """
    input_train = pd.read_csv(TRAIN_LOC, index_col = 'PassengerId')
    input_test = pd.read_csv(TEST_LOC, index_col = 'PassengerId')

    return input_train, input_test


def write_output(y_test, input_test):
    """
    writes the output as a csv file
    :param y_test: array, prediction results (0 or 1) for all indices
    :param input_test:, array, used to get the indices
    :return:
    """
    y_test = pd.DataFrame({"PassengerId" : input_test.index,
                           "Survived" : y_test})
    y_test.to_csv(path_or_buf=WRITE_LOC, index=False)


def plot_bar(values, labels, title):
    """
    Visualisation of data
    :param values: list of scalars, heights of the data bars
    :param labels: list of strings, labels of the data bars
    :param title: string, title for the window
    :return:
    """
    plt.bar([x for x in range(len(values))], values, tick_label=labels)
    plt.title(title)
    plt.show()



def main():
    input_train, input_test = read_input()

    # number of unique values in the columns
    uniqs = input_train.nunique()
    # number of null values in the columns
    nulls = pd.isna(input_train).sum()

    plot_bar(uniqs, input_train.columns, "Uniques")
    print("Unique values by column \n", uniqs, "\n")

    plot_bar(nulls, input_train.columns, "Null values")
    print("Null values by column \n", nulls, "\n")

    # "Survived" column of the data, the prediction target
    y_train_all = input_train["Survived"]
    # Drop the "Survived" column from the trainig data
    # and columns that have too many unique or null values to help with predictions
    X_train_all = input_train.drop(columns=["Survived", "Name", "Ticket", "Cabin"], axis=1)

    X_test = input_test.drop(columns=["Name", "Ticket", "Cabin"], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=0)

    # fill the missing values
    nimp = SimpleImputer(strategy="median")

    # columns with numerical data
    num_cols = X_train.select_dtypes(exclude="object").columns
    # columns with non-numerical data
    cat_cols = X_train.select_dtypes(include="object").columns

    cimp = SimpleImputer(strategy="most_frequent")
    ohe = OneHotEncoder()

    # Transforms columns with non-numerical data
    cat_transformer = Pipeline(steps=[
        ("impute", cimp),
        ("encode", ohe)
    ])

    # Transform numerical and non-numerical columns separately
    col_trsfmr = ColumnTransformer(transformers=[
        ("num_vars", nimp, num_cols),
        ("cat_vars", cat_transformer, cat_cols)
    ],
    n_jobs=THREADS)


    # test these classifiers
    ensemble_classifiers = [
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        AdaBoostClassifier()
    ]

    best_score = [0, 0, 0]

    # grid search the best parameters
    for clf in ensemble_classifiers:
        clf.n_jobs = THREADS
        clf.random_state = 1

        for n_ests in range(20, 250, 10):
            clf.n_estimators = n_ests

            classifier = Pipeline(steps=[
                ("preprocess", col_trsfmr),
                ("classify", clf)
            ])

            classifier.fit(X_train, y_train)

            y_pred = classifier.predict(X_valid)
            score = precision_score(y_valid, y_pred)

            if score > best_score[0]:
                best_score[0] = score
                best_score[1] = n_ests
                best_score[2] = deepcopy(clf)

    print(best_score)


    # Use the best classifier to predict the final results
    best_clf = best_score[2]

    best_clf = Pipeline(steps=[
        ("preprocess", col_trsfmr),
        ("classify", best_clf)
    ])

    # fit the best classifier with all available training data
    best_clf.fit(X_train_all, y_train_all)

    # write the output to a csv file
    y_test = best_clf.predict(X_test)

    write_output(y_test, input_test)



main()