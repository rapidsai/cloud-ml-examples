#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
    multi-GPU and multi-CPU HPO workflow
    includes data loading, splitting, model training, and scoring/inference
"""

import sys, os, time, logging, json, pprint, argparse
import warnings; warnings.simplefilter("ignore", (UserWarning, FutureWarning))

# shared imports
from dask.distributed import wait, Client, LocalCluster
from dask_ml.model_selection import train_test_split
import xgboost, numpy as np

# CPU imports
try: 
    import dask, sklearn, pandas
    from sklearn.metrics import accuracy_score as sklearn_accuracy_score
    
except: print( ' unable to load CPU libraries ')

# GPU imports
try:
    import cudf, dask_cudf, cuml, cupy
    from dask_cuda import LocalCUDACluster
    from cuml.dask.common.utils import persist_across_workers    
    from cuml.metrics import accuracy_score as cuml_accuracy_score

except: print( ' unable to load GPU libraries ')

default_sagemaker_paths = {
    'base' : '/opt/ml',
    'code' : '/opt/ml/code',
    'data' : '/opt/ml/input',
    'train_data' : '/opt/ml/input/data/training',
    'hyperparams' : '/opt/ml/input/config/hyperparameters.json',
    'model' : '/opt/ml/model',
    'output' : '/opt/ml/output' }

class RapidsCloudML ( object ):

    def __init__ ( self, 
                   model_type = 'RandomForest', 
                   compute_type = 'multi-GPU',
                   CSP_paths = default_sagemaker_paths ):

        self.CSP_paths = CSP_paths
        self.model_type = model_type
        self.compute_type = compute_type
        
        # CPU or GPU cluster
        if 'multi-GPU' in self.compute_type:
            self.n_workers = cupy.cuda.runtime.getDeviceCount()
            self.cluster = LocalCUDACluster( n_workers = self.n_workers )
            self.client = Client( self.cluster )
            print(f'dask multi-GPU cluster with {self.n_workers} workers ')
            
        elif 'multi-CPU' in self.compute_type:
            self.n_workers = os.cpu_count()
            self.cluster = LocalCluster(  n_workers = self.n_workers, threads_per_worker = 1 )
            self.client = Client( self.cluster )
            print(f'dask multi-CPU cluster with {self.n_workers} workers')
        else:
            self.cluster = None; self.client = None
        
    def load_data ( self, filename = '*.parquet', 
                    columns = None  ):

        target_filename = self.CSP_paths['train_data'] + '/' + filename
        self.log( f'\n> loading dataset from {target_filename}...\n')

        with PerfTimer ( self, 'ingestion_timer'):
            if 'multi-CPU' in self.compute_type:
                dataset = dask.dataframe.read_parquet( target_filename, columns = columns )
                
            elif 'multi-GPU' in self.compute_type:
                dataset = dask_cudf.read_parquet( target_filename, columns = columns )
                
            dataset = dataset.dropna()
            dataset = dataset.repartition( npartitions = self.n_workers * 4 )
        
        print(f'dataset len : {len(dataset)}')
        return dataset
   
    def split_data ( self, dataset, y_label, train_size = .8, random_state = 0, shuffle = True  ) :

        with PerfTimer( self, 'split_timer'):
            train, test = train_test_split( dataset, random_state = random_state ) # unable to shuffle -- no dask_cudf sampler implemented
            
            X_train, y_train = train.drop(y_label, axis = 1).astype('float32'), train[y_label].astype('int32')
            X_test, y_test = test.drop(y_label, axis = 1).astype('float32'), test[y_label].astype('int32')
        
        if 'multi-GPU' in self.compute_type:
            with PerfTimer( self, 'persist_timer'):
                workers = self.client.has_what().keys()
                X_train, X_test, y_train, y_test = persist_across_workers( self.client,
                                                                           [X_train, X_test, y_train, y_test],
                                                                           workers = workers )
                wait( [X_train, X_test, y_train, y_test] );

        return X_train, X_test, y_train, y_test

    def train_model ( self, X_train, y_train, model_params ):
        
        with PerfTimer(self, 'train_timer'):

            if 'XGBoost' in self.model_type:
                dtrain = xgboost.dask.DaskDMatrix( self.client, X_train, y_train)
                
                # avoids warning messages
                boosting_rounds = model_params.pop('num_boost_round')                
                
                trained_model = xgboost.dask.train( self.client, model_params, dtrain, 
                                                    num_boost_round = boosting_rounds )
                return trained_model['booster']
                
            elif 'RandomForest' in self.model_type:
                if 'GPU' in self.compute_type:            
                    from cuml.dask.ensemble import RandomForestClassifier
                    rf_model = RandomForestClassifier ( n_estimators = model_params['n_estimators'],
                                                        max_depth = model_params['max_depth'],                                                
                                                        max_features = model_params['max_features'],
                                                        n_bins = 32 )
                else:
                    from sklearn.ensemble import RandomForestClassifier
                    rf_model = RandomForestClassifier ( n_estimators = model_params['n_estimators'],
                                                        max_depth = model_params['max_depth'],                                                
                                                        max_features = model_params['max_features'], 
                                                        n_jobs=-1 )
                    
                trained_model = rf_model.fit( X_train, y_train)
                return trained_model
            print(len(X_train))
        return None
    
    def evaluate_test_perf ( self, trained_model, X_test, y_test, threshold = 0.5 ):
        with PerfTimer(self, 'score_timer'):
            
            if 'XGBoost' in self.model_type:                
                dtest = xgboost.dask.DaskDMatrix( self.client, X_test, y_test)
                predictions = xgboost.dask.predict( self.client, trained_model, dtest).compute()
                predictions = np.where( predictions >= threshold, 1, 0) # threshold returned probabilities into 0/1 labels
                
            elif 'RandomForest' in self.model_type:
                predictions = trained_model.predict( X_test )
                if 'multi-CPU' not in self.compute_type:
                    predictions = predictions.compute()
            
            if 'multi' in self.compute_type:
                y_test = y_test.compute()

            if 'GPU' in self.compute_type:                
                test_accuracy = cuml_accuracy_score ( y_test, predictions )
            elif 'CPU' in self.compute_type:
                test_accuracy = sklearn_accuracy_score ( y_test, predictions )
                    
        # accumulate internal list
        return test_accuracy
    
    # emit score so sagemaker can parse it (using string REGEX)
    def emit_score ( self, test_accuracy ):
        self.log( f'\n\t test-accuracy: {test_accuracy}; \n')

    def save_best_model ( self, global_best_model = None ):
        pass

    def set_up_logging( self ):        
        logging_path = self.CSP_paths['output'] + '/log.txt'
        logging.basicConfig( filename= logging_path,
                             level=logging.INFO)

    def log ( self, text ):
        logging.info( text )
        print( text )
    

#  parse ML model parameters [ e.g., passed in by cloud HPO ]
def parse_model_parameters ( input_args, config_params ):
    print('parsing model hyper-parameters from command line arguments...\n')
    parser = argparse.ArgumentParser ()

    if 'XGBoost' in config_params['model_type']:
        # xgboost - supported hyperparameters
        parser.add_argument( '--max_depth',       type = int,   default = 6,    
                             help = 'maximum tree depth' )
        parser.add_argument( '--num_boost_round', type = int,   default = 100,  
                             help = 'number of boosted trees' )
        parser.add_argument( '--learning_rate',   type = float, default = 0.3,  
                             help = 'step size shrinkage between rounds, prevents overfitting' )    
        parser.add_argument( '--gamma',           type = float, default = 0.,   
                             help = 'minimum loss reduction required to make a leaf node split, prevents overfitting' )
        parser.add_argument( '--lambda_',         type = float, default = 1.,   
                             help = 'L2 regularizaiton term on weights, prevents overfitting' )
        parser.add_argument( '--alpha',           type = float, default = 0.,   
                             help = 'L1 regularizaiton term on weights, prevents overfitting' )
        
        args, unknown_args = parser.parse_known_args( input_args )
        
        model_params = {            
            'max_depth' : args.max_depth,
            'num_boost_round': args.num_boost_round,
            'learning_rate': args.learning_rate,
            'gamma': args.gamma,
            'random_state' : 0,
        }        
        if 'GPU' in config_params['compute']:
            model_params.update( { 'tree_method': 'gpu_hist' })
        else:
            model_params.update( { 'tree_method': 'hist' })

    elif 'RandomForest' in config_params['model_type']:

        # random-forest - supported hyperparameters
        parser.add_argument( '--n_estimators', type = int,   default = 100,  
                             help = 'number of trees in a random forest' )
        parser.add_argument( '--max_depth',    type = int,   default = 6,  
                             help = 'maximum tree depth' )
        parser.add_argument( '--max_features', type = float, default = .5, 
                             help = 'ratio of the number of features to consider per node split' )

        args, unknown_args = parser.parse_known_args( input_args )

        # random forest
        model_params = {            
            'max_depth' : args.max_depth,
            'n_estimators' : args.n_estimators,        
            'max_features': args.max_features,
            'seed' : 0,        
        }
    else:
        raise Exception(f"!error: unknown model type {config_params['model_type']}")

    pprint.pprint( model_params, indent = 5 ); print( '\n' )
    return model_params

# use job name to define model type, compute, data
def parse_job_name( config = None ):
    print('\nparsing job config from job filename...\n')    
    try:
        if 'SM_TRAINING_ENV' in os.environ:
            env_params = json.loads( os.environ['SM_TRAINING_ENV'] )
            job_name = env_params['job_name']

            # parse model type [ first element of job name ]
            if 'rf' in job_name.split('-')[0].lower():
                config['model_type'] = 'RandomForest'
            elif 'xgb' in job_name.split('-')[0].lower():
                config['model_type'] = 'XGBoost'
            else:
                pass
                # raise Exception( ' unsupported model type ')

            # compute
            if 'mgpu' in job_name.split('-')[1].lower():
                config['compute'] = 'multi-GPU'
            elif 'mcpu':
                config['compute'] = 'multi-CPU'

            # parse CV folds
            config['CV_folds'] = int(job_name.split('-')[3])

            assert(config['CV_folds'] > 0 )

        else:
            print( 'unable to parse config from job filename, loading defaults...\n')        

    except Exception as error:
        print( error )

    pprint.pprint( config, indent = 5 ); print( '\n' )
    return config


# perf_counter = highest available timer resolution 
class PerfTimer:
    def __init__(self, rcml, name_string = '' ):
        self.start = None
        self.rcml = rcml
        self.duration = None
        self.name_string = name_string

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):        
        self.duration = time.perf_counter() - self.start
        self.rcml.log(f"|-> {self.name_string} : {self.duration:.4f}\n")