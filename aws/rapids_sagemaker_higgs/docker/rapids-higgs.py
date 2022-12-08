                                  #!/usr/bin/env python
# coding: utf-8

from cuml import RandomForestClassifier as cuRF
from cuml.preprocessing.model_selection import train_test_split
import cudf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os
from urllib.request import urlretrieve
import gzip
import argparse


def main(args):
    
    # SageMaker options
    model_dir       = args.model_dir
    data_dir        = args.data_dir
        
    col_names = ['label'] + ["col-{}".format(i) for i in range(2, 30)] # Assign column names
    dtypes_ls = ['int32'] + ['float32' for _ in range(2, 30)] # Assign dtypes to each column
    
    data = cudf.read_csv(data_dir+'HIGGS.csv', names=col_names, dtype=dtypes_ls)
    X_train, X_test, y_train, y_test = train_test_split(data, 'label', train_size=0.70)

    # Hyper-parameters
    hyperparams={
    'n_estimators'       : args.n_estimators,
    'max_depth'          : args.max_depth,
    'n_bins'             : args.n_bins,
    'split_criterion'    : args.split_criterion,
    'bootstrap'          : args.bootstrap,
    'max_leaves'         : args.max_leaves,
    'max_features'       : args.max_features
    }

    cu_rf = cuRF(**hyperparams)
    cu_rf.fit(X_train, y_train)

    print("test_acc:", accuracy_score(cu_rf.predict(X_test), y_test)

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # Hyper-parameters
    parser.add_argument('--n_estimators',        type=int,   default=20)
    parser.add_argument('--max_depth',           type=int,   default=16)
    parser.add_argument('--n_bins',              type=int,   default=8)
    parser.add_argument('--split_criterion',     type=int,   default=0)
    parser.add_argument('--bootstrap',           type=bool,  default=True)
    parser.add_argument('--max_leaves',          type=int,   default=-1)
    parser.add_argument('--max_features',        type=float, default=0.2)
    
    # SageMaker parameters
    parser.add_argument('--model_dir',        type=str)
    parser.add_argument('--model_output_dir', type=str,   default='/opt/ml/output/')
    parser.add_argument('--data_dir',         type=str,   default='/opt/ml/input/data/dataset/')
    
    args = parser.parse_args()
    main(args)


