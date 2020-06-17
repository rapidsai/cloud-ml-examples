import argparse
from functools import partial

import mlflow
import mlflow.sklearn

from cuml.metrics.accuracy import accuracy_score as cuml_acc
from cuml.preprocessing.model_selection import train_test_split as cuml_split
from cuml.ensemble import RandomForestClassifier as cu_RF

from sklearn.model_selection import train_test_split as sk_split
from sklearn.metrics import accuracy_score as sk_acc
from sklearn.ensemble import RandomForestClassifier as sk_RF
import pandas as pd

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


def load_data(fpath, compute_type):
    """
    Simple helper function for loading data to be used by CPU/GPU models.

    :param fpath: Path to the data to be ingested
    :param compute_type: [CPU|GPU]
    :return: DataFrame wrapping the data at [fpath]. Data will be in either a Pandas or RAPIDS (cuDF) DataFrame
    """
    if 'CPU' in compute_type:
        try:
            import pandas
            import pyarrow
            from pyarrow import orc
        except Exception as error:
            print(f'Failed to import pandas and pyarrow: {error}')
    elif 'GPU' in compute_type:
        try:
            import cudf
        except Exception as error:
            print(f'Failed to import cuDF modules: {error}')

    if 'CPU' in compute_type:
        df = pd.read_parquet(fpath)
    else:
        df = cudf.read_parquet(fpath)

    return df


def _train(params, fpath, mode='GPU', hyperopt=False):
    """
    :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
    :param fpath: Path or URL for the training data used with the model.
    :param mode: Hardware backend to use for training [CPU|GPU]
    :param hyperopt: Use hyperopt for hyperparameter search during training.
    :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
    """
    max_depth, max_features, n_estimators = params
    max_depth, max_features, n_estimators = int(max_depth), float(max_features), int(n_estimators)

    df = load_data(fpath, compute_type=mode)

    X = df.drop(["ArrDelayBinary"], axis=1)
    y = df["ArrDelayBinary"].astype('int32')

    if mode == "GPU":
        X_train, X_test, y_train, y_test = cuml_split(X, y, test_size=0.2)
        mod = cu_RF(max_depth=max_depth, max_features=max_features, n_estimators=n_estimators)
        acc_scorer = cuml_acc
    elif mode == "CPU":
        X_train, X_test, y_train, y_test = sk_split(X, y, test_size=0.2)
        mod = sk_RF(max_depth=max_depth, max_features=max_features, n_estimators=n_estimators)
        acc_scorer = sk_acc
    else:
        raise RuntimeError("Unknown option. Choose between [CPU, GPU], case sensitive.")

    mod.fit(X_train, y_train)
    preds = mod.predict(X_test)
    acc = acc_scorer(y_test, preds)

    mlparams = {"max_depth": str(max_depth),
                "max_features": str(max_features),
                "n_estimators": str(n_estimators),
                "mode": str(mode)}
    mlflow.log_params(mlparams)

    mlmetrics = {"accuracy": acc}
    mlflow.log_metrics(mlmetrics)

    if (not hyperopt):
        return mod

    return {'loss': acc, 'status': STATUS_OK}


def train(params, fpath, mode='GPU', hyperopt=False):
    """
    Proxy function used to call _train
    :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
    :param fpath: Path or URL for the training data used with the model.
    :param mode: Hardware backend to use for training [CPU|GPU]
    :param hyperopt: Use hyperopt for hyperparameter search during training.
    :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
    """
    with mlflow.start_run(nested=True):
        return _train(params, fpath, mode, hyperopt)


if (__name__ in ("__main__",)):
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='tpe', choices=['tpe'], type=str)
    parser.add_argument('--conda-env', required=True, type=str)
    parser.add_argument('--fpath', required=True, type=str)
    parser.add_argument('--mode', default='GPU', choices=['GPU', 'CPU'], type=str)
    args = parser.parse_args()

    search_space = [
        hp.uniform('max_depth', 5, 20),
        hp.uniform('max_features', 0., 1.0),
        hp.uniform('n_estimators', 150, 1000)
    ]

    trials = Trials()
    algorithm = tpe.suggest if args.algo == 'tpe' else None
    fn = partial(train, fpath=args.fpath, mode=args.mode, hyperopt=True)
    experid = 0

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "RAPIDS-Hyperopt-Databricks")
        argmin = fmin(fn=fn,
                      space=search_space,
                      algo=algorithm,
                      max_evals=2,
                      trials=trials)

        print("===========")
        fn = partial(train, fpath=args.fpath, mode=args.mode, hyperopt=False)
        final_model = fn(tuple(argmin.values()))

        conda_data = ""
        if (args.conda_env.startswith("http")):
            import requests

            resp = requests.get(args.conda_env)
            conda_data = str(resp.text)
        else:
            with open(args.conda_env, 'r') as reader:
                conda_data = reader.read()

        with open("conda.yaml", 'w') as writer:
            writer.write(conda_data)

        mlflow.sklearn.log_model(final_model, "cuml_rf_test", conda_env='conda.yaml')
