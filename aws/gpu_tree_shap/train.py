from __future__ import print_function

import argparse
import json
import logging
import os
import pickle as pkl

import pandas as pd
import xgboost as xgb
from sagemaker_containers import entry_point
from sagemaker_xgboost_container import distributed
# from sagemaker_xgboost_container.data_utils import get_dmatrix
import time

import shutil
import csv


# Heavily draws from AWS's data_utils.py script: 
# https://github.com/aws/sagemaker-xgboost-container/blob/cf2a05a525606225e91d7e588f57143827cbe3f7/src/sagemaker_xgboost_container/data_utils.py
# but we use our own version because the AWS version does not allow you to specify the label column.
def get_dmatrix(data_path, content_type, label_column, csv_weights=0):
    """Create Data Matrix from CSV file.
    Assumes that sanity validation for content type has been done.
    :param data_path: Either directory or file
    :param content_type: Only supports "csv"
    :param label_column: Integer corresponding to index of the label column, starting with 0
    :param csv_weights: 1 if the instance weights are in the second column of csv file; otherwise, 0
    :return: xgb.DMatrix or None
    """
    
    if "csv" not in content_type.lower():
        raise Exception("File type '{}' not supported".format(content_type))

    if not isinstance(data_path, list):
        if not os.path.exists(data_path):
            logging.info('File path {} does not exist!'.format(data_path))
            return None
        files_path = get_files_path(data_path)
    else:
        # Create a directory with symlinks to input files.
        files_path = "/tmp/sagemaker_xgboost_input_data"
        shutil.rmtree(files_path, ignore_errors=True)
        os.mkdir(files_path)
        for path in data_path:
            if not os.path.exists(path):
                return None
            if os.path.isfile(path):
                os.symlink(path, os.path.join(files_path, os.path.basename(path)))
            else:
                for file in os.scandir(path):
                    os.symlink(file, os.path.join(files_path, file.name))

    dmatrix = get_csv_dmatrix(files_path, label_column, csv_weights)

    return dmatrix


def get_files_path(data_path):
    if os.path.isfile(data_path):
        files_path = data_path
    else:
        for root, dirs, files in os.walk(data_path):
            if dirs == []:
                files_path = root
                break

    return files_path


def get_csv_dmatrix(files_path, label_column, csv_weights):
    """Get Data Matrix from CSV data in file mode.
    Infer the delimiter of data from first line of first data file.
    :param files_path: File path where CSV formatted training data resides, either directory or file
    :param label_column: Integer corresponding to index of the label column, starting with 0
    :param csv_weights: 1 if instance weights are in second column of CSV data; else 0
    :return: xgb.DMatrix
    """
    csv_file = files_path if os.path.isfile(files_path) else [
        f for f in os.listdir(files_path) if os.path.isfile(os.path.join(files_path, f))][0]
    with open(os.path.join(files_path, csv_file)) as read_file:
        sample_csv_line = read_file.readline()
    delimiter = _get_csv_delimiter(sample_csv_line)

    try:
        if csv_weights == 1:
            dmatrix = xgb.DMatrix(
                '{}?format=csv&label_column={}&delimiter={}&weight_column=1'.format(files_path, label_column, delimiter))
        else:
            dmatrix = xgb.DMatrix('{}?format=csv&label_column={}&delimiter={}'.format(files_path, label_column, delimiter))

    except Exception as e:
        raise Exception("Failed to load csv data with exception:\n{}".format(e))

    return dmatrix


def _get_csv_delimiter(sample_csv_line):
    try:
        delimiter = csv.Sniffer().sniff(sample_csv_line).delimiter
        logging.info("Determined delimiter of CSV input is \'{}\'".format(delimiter))
    except Exception as e:
        raise Exception("Could not determine delimiter on line {}:\n{}".format(sample_csv_line[:50], e))
    return delimiter


def _xgb_train(params, dtrain, evals, num_boost_round, model_dir, is_master):
    """Run xgb train on arguments given with rabit initialized.

    This is our rabit execution function.

    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training,
                        or is running single node training job.
                        Note that rabit_run will include this argument.
    """
    
    start = time.time()
    booster = xgb.train(params=params, dtrain=dtrain, evals=evals, num_boost_round=num_boost_round)
    logging.info("XGBoost training time {}".format(time.time() - start))

    if is_master:
        model_location = model_dir + "/xgboost-model"
        pkl.dump(booster, open(model_location, "wb"))
        logging.info("Stored trained model at {}".format(model_location))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument(
        "--max_depth",
        type=int,
    )
    parser.add_argument("--eta", type=float)
    parser.add_argument("--gamma", type=int)
    parser.add_argument("--min_child_weight", type=int)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--verbosity", type=int)
    parser.add_argument("--objective", type=str)
    parser.add_argument("--num_round", type=int)
    parser.add_argument("--tree_method", type=str, default="gpu_hist")  # "auto", "hist", or "gpu_hist"
    parser.add_argument("--predictor", type=str, default="gpu_predictor")  # "auto"
    
    # e.g., 'sklearn.datasets.fetch_california_housing()'
    parser.add_argument("--sklearn_dataset", type=str, default="None")
    # specify file type
    parser.add_argument("--content_type", type=str)
    # if csv, should specify a label column
    parser.add_argument("--label_column", type=int)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--sm_hosts", type=str, default=os.environ.get("SM_HOSTS"))
    parser.add_argument("--sm_current_host", type=str, default=os.environ.get("SM_CURRENT_HOST"))

    args, _ = parser.parse_known_args()

    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host

    sklearn_dataset = args.sklearn_dataset
    if "None" in sklearn_dataset: 
        dtrain = get_dmatrix(args.train, args.content_type, args.label_column)
        try:
            dval = get_dmatrix(args.validation, args.content_type, args.label_column)
        except Exception: 
            dval = None
    else:  # Use a dataset from sklearn.datasets
        import sklearn.datasets
        try:
            # e.g., sklearn_dataset = "sklearn.datasets.fetch_california_housing()"
            data = eval(sklearn_dataset)
        except Exception: 
            raise ValueError("Function {} is not supported. Try something like 'sklearn.datasets.fetch_california_housing()'"
                             .format(sklearn_dataset))
                
        X = data.data
        y = data.target
        dtrain = xgb.DMatrix(X, y)
        dval = None
        
        
    watchlist = (
        [(dtrain, "train"), (dval, "validation")] if dval is not None else [(dtrain, "train")]
    )

    train_hp = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "verbosity": args.verbosity,
        "objective": args.objective,
        "tree_method": args.tree_method,
        "predictor": args.predictor,
    }

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        model_dir=args.model_dir,
    )

    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()

        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=_xgb_train,
            args=xgb_train_args,
            include_in_training=(dtrain is not None),
            hosts=sm_hosts,
            current_host=sm_current_host,
            update_rabit_args=True,
        )
    else:
        # If single node training, call training method directly.
        if dtrain:
            xgb_train_args["is_master"] = True
            _xgb_train(**xgb_train_args)
        else:
            raise ValueError("Training channel must have data to train model.")


def model_fn(model_dir):
    """Deserialize and return fitted model.

    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    model_file = "xgboost-model"
    booster = pkl.load(open(os.path.join(model_dir, model_file), "rb"))
    return booster
