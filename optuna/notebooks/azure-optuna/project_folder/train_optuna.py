import random
import time
from contextlib import contextmanager

import cudf
import cuml
import dask_cudf
import numpy as np
import optuna
import pandas as pd
import sklearn
import os
import dask
import dask_optuna

from cuml import LogisticRegression
from cuml.preprocessing.model_selection import train_test_split
from cuml.metrics import log_loss

from dask_cuda import LocalCUDACluster
from dask.distributed import Client, wait, performance_report

from joblib import parallel_backend
import argparse
from azureml.core.run import Run

run = Run.get_context()

def train_and_eval(X_param, y_param, penalty='l2', C=1.0, l1_ratio=None, fit_intercept=True):
    """
        Splits the given data into train and test split to train and evaluate the model
        for the params parameters.
        
        Params
        ______
        
        X_param:  DataFrame. 
                  The data to use for training and testing. 
        y_param:  Series. 
                  The label for training
        penalty, C, l1_ratio, fit_intercept: The parameter values for Logistic Regression.

        Returns
        score: log loss of the fitted model
    """
    X_train, X_valid, y_train, y_valid = train_test_split(X_param,
                                                          y_param,
                                                          random_state=42)
    classifier = LogisticRegression(penalty=penalty,
                                    C=C,
                                    l1_ratio=l1_ratio,
                                    fit_intercept=fit_intercept)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_valid)
    score = log_loss(y_valid, y_pred)
    return score

def objective(trial, X_param, y_param):
    """
    Objective function used for Optuna experiments.
    Returns the score for optimization task.
    """
    C = trial.suggest_uniform("C", 0 , 9.0)
    penalty = trial.suggest_categorical("penalty", ['l1', 'none', 'l2'])
    l1_ratio = trial.suggest_uniform("l1_ratio", 0 , 1.0)
    fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])

    score = train_and_eval(X_param,
                           y_param,
                           penalty=penalty,
                           C=C,
                           l1_ratio=l1_ratio,
                           fit_intercept=fit_intercept)
    run.log('C', np.float(C))
    run.log('penalty', penalty)
    run.log('l1_ratio', np.float(l1_ratio))
    run.log('fit_intercept', fit_intercept)
    run.log('score',np.float(score))
    return score
if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, help='location of data')
    args = parser.parse_args()
    data_dir = args.data_dir
    print('Data folder is at:', data_dir)
    print('List all files: ', os.listdir(data_dir))

    cluster = LocalCUDACluster(threads_per_worker=1, ip="", dashboard_address="8088")
    c = Client(cluster)

    # Query the client for all connected workers
    workers = c.has_what().keys()
    n_workers = len(workers)
    df = cudf.read_csv(os.path.join(data_dir, "train.csv"))
    N_TRIALS = 5

    # Drop non-numerical data and fill NaNs before passing to cuML RF
    CAT_COLS = list(df.select_dtypes('object').columns)
    df = df.drop(CAT_COLS, axis=1)
    df = df.dropna()

    df = df.astype("float32")
    X, y = df.drop(["target"], axis=1), df["target"].astype('int32')

    study_name = "dask_optuna_lr_log_loss_tpe"
    storage_name = "sqlite:///study_stores.db"

    storage = dask_optuna.DaskStorage(storage_name)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                study_name=study_name,
                                direction="minimize",
                                storage=storage)
    # Optimize in parallel on your Dask cluster
    with parallel_backend("dask"):
        study.optimize(lambda trial: objective(trial, X, y),
                           n_trials=N_TRIALS,
                           n_jobs=n_workers)
    print('Best params{} and best score{}'.format(study.best_params, study.best_value))