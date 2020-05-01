import argparse
import os
import time
from urllib.request import urlretrieve

#importing necessary libraries
import numpy as np
import pandas as pd

import cudf
import cuml
from cuml import RandomForestClassifier as cuRF
from cuml.preprocessing.model_selection import train_test_split
from cuml.metrics.accuracy import accuracy_score

import rapids_csp_azure

from azureml.core.run import Run
run = Run.get_context()

def main():
    
    rcml = rapids_csp_azure.RapidsCloudML(cloud_type = 'Azure', model_type = 'RandomForest')

    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees in RF')
    parser.add_argument('--max_depth', type=int, default=16, help='Max depth of each tree')
    parser.add_argument('--n_bins', type=int, default=8, help='Number of bins used in split point calculation')
    parser.add_argument('--max_features', type=float, default=1.0, help='Number of features for best split')


    args = parser.parse_args()
    data_dir = args.data_dir

#     df = cudf.read_parquet(os.path.join(data_dir, 'airline_1000.parquet'))
   
#     # Encode categoricals as numeric
#     for col in df.select_dtypes(['object']).columns:
#         df[col] = df[col].astype('category').cat.codes.astype(np.float32)
        
#     # cast all remaining columns to float32
#     for col in df.columns:
#         if col in df.select_dtypes(['object']).columns: pass
#         else:
#             df[col] = df[col].astype(np.float32)
            
#     df['ArrDelayBinary'] = 1 * (df['ArrDelay'] > 10)
    
     # ingest data [ post pre-processing ]
    filename = os.path.join(data_dir, 'airline_20m.parquet')
    dataset, col_labels, y_label, ingest_time = rcml.load_data ( filename = filename )
    
    X = dataset[dataset.columns.difference(['ArrDelay', 'ArrDelayBinary'])]
    y = dataset['ArrDelayBinary'].astype(np.int32)
    del dataset
    
    X_train, X_test, y_train, y_test, split_time = rcml.split_data(X, y, random_state =77, shuffle = True )

    n_estimators = args.n_estimators
    run.log('n_estimators', np.int(args.n_estimators))
    max_depth = args.max_depth
    run.log('max_depth', np.int(args.max_depth))
    n_bins = args.n_bins
    run.log('n_bins', np.int(args.n_bins))
    max_features = args.max_feature   
    run.log('max_features', np.str(args.max_features))
    
    model_params = {
        'n_estimators' : n_estimators,
        'max_depth' : max_depth,
        'max_features': max_features,
        'n_bins' : n_bins,
    }
        
    print('\n---->>>> Training on GPUs <<<<----\n')
        
    trained_model, training_time = rcml.train_model (X_train, y_train, model_params)

    test_accuracy, infer_time = rcml.evaluate_test_perf (trained_model, X_test, y_test)
        
    print('Exiting script')
    

if __name__ == '__main__':
    main()
