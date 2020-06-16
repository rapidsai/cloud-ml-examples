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
    data_type = 'ORC' if fpath.lower().endswith('.orc') else 'parquet'

    if 'CPU' in compute_type:
        try:
            import pandas
            import pyarrow
            from pyarrow import orc
        except Exception as error:
            print(f'! CPU import error : {error}')
    elif 'GPU' in compute_type:
        try:
            import cudf
        except Exception as error:
            print(f'! GPU import error : {error}')

    if 'CPU' in compute_type:
        if ('ORC' == data_type):
            if (fpath.startswith('/') or fpath.startswith('file:')):
                with open(fpath, mode='rb') as reader:
                    df = pyarrow.orc.ORCFile(reader).read().to_pandas()
            else:
                raise NotImplemented("CPU Remote read not implemented")
        else:
            if (fpath.startswith('/') or fpath.startswith('file:')):
                df = pd.read_parquet(fpath)
            else:
                raise NotImplemented("CPU Remote read not implemented")


    elif ('GPU' in compute_type):
        if ('ORC' == data_type):
            df = cudf.read_orc(fpath)
        else:
            df = cudf.read_parquet(fpath)

    return df


def _train(params, fpath, mode='GPU', log_to_mlflow=None, hyperopt=False):
    max_depth, max_features, n_estimators = params
    max_depth, max_features, n_estimators = int(max_depth), float(max_features), int(n_estimators)

    df = load_data(fpath, compute_type=mode)

    for col in df.columns:
        df[col] = df[col].astype('float32')

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

    if (log_to_mlflow):
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


def train(params, fpath, mode='GPU', log_to_mlflow=None, hyperopt=False):
    """
    An example train method that computes the square of the input.
    This method will be passed to `hyperopt.fmin()`.

    :param params: hyperparameters. Its structure is consistent with how search space is defined. See below.
    :return: dict with fields 'loss' (scalar loss) and 'status' (success/failure status of run)
    """
    if (log_to_mlflow):
        with mlflow.start_run(nested=True):
            return _train(params, fpath, mode, log_to_mlflow, hyperopt)
    else:
        return _train(params, fpath, mode, log_to_mlflow, hyperopt)


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
    fn = partial(train, fpath=args.fpath, mode=args.mode, log_to_mlflow=True, hyperopt=True)
    experid = 0

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", "RAPIDS-Hyperopt-Databricks")
        argmin = fmin(fn=fn,
                      space=search_space,
                      algo=algorithm,
                      max_evals=2,
                      trials=trials)

        print("===========")
        fn = partial(train, fpath=args.fpath, mode=args.mode, log_to_mlflow=False, hyperopt=False)
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
