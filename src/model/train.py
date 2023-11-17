# Import libraries

import argparse
import glob
import os

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mlflow.sklearn


# define functions to call from outside
def train(training_data, reg_rate):
    # read data
    df = get_csvs_df(training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(reg_rate, X_train, X_test, y_train, y_test)



def main(args):
    # TO DO: enable autologging
    mlflow.sklearn.autolog()

    # train model
    train(args.training_data, args.reg_rate)

def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data
def split_data(df):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df, 0.8, random_state=1)
    # Return the split data
    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
