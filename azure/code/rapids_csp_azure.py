# os
import sys, os, time, logging

# CPU DS stack 
import pandas as pd
import numpy as np
import sklearn

# GPU DS stack [ rapids ]
import cudf, cuml

# scaling library
import dask

# ML models
from sklearn import ensemble
import xgboost

# data set splits
from cuml.preprocessing.model_selection import train_test_split as cuml_train_test_split
from sklearn.model_selection import train_test_split as sklearn_train_test_split

# device query
import cupy

# memory query
import psutil
import pynvml

# i/o
import logging, json, pprint

default_azureml_paths = {
    'train_script' : './train_script',
#     'base' : '/opt/ml',
#     'code' : '/opt/ml/code',
#     'data' : '/opt/ml/input',
    'train_data' : './data_airline',
#     'hyperparams' : '/opt/ml/input/config/hyperparameters.json',
#     'model' : '/opt/ml/model',
    'output' : './output',
}

CV_FOLDS = 5

class RapidsCloudML ( object ):

    def __init__ ( self, cloud_type = 'Azure', 
                   model_type = 'RandomForest', 
                   data_type = 'Parquet',
                   compute_type = 'single-GPU', 
#                    n_workers = -1, 
                   verbose_estimator = False,
                   CSP_paths = default_azureml_paths):

        self.CSP_paths = CSP_paths
        self.cloud_type = cloud_type        
        self.model_type = model_type
        self.data_type = data_type
        self.compute_type = compute_type
        self.verbose_estimator = verbose_estimator
#         self.n_workers = self.parse_compute( n_workers )
        # self.query_memory()

    def load_hyperparams( self, model_name = 'XGBoost', CV_folds = CV_FOLDS ):
            self.log_to_file('\n> loading hyperparameters \n')

            if self.model_type == 'XGBoost':
                # https://xgboost.readthedocs.io/en/latest/parameter.html
                model_params = { 
                    'max_depth': 6,                     # default = 6             :: maximum depth of a tree
                    'num_boost_round': 100,             # default = XXX           :: number of trees        
                    'learning_rate': 0.3,               # default = 0.3           :: step size shrinkage between rounds, prevents overfitting
                    'gamma': 0.,                        # default = 0             :: minimum loss reduction required to make a leaf node split, prevents overfitting
                    'lambda': 1.,                       # default = 1             :: L2 regularizaiton term on weights, prevents overfitting
                    'alpha': 0.,                        # default = 0             :: L1 regularization term on weights, prevents overfitting
                    'tree_method': 'gpu_hist',          # default = 'gpu_hist'    :: tree construction algorithm
                    'random_state' : 0
                }
            elif self.model_type == 'RandomForest':
                # https://docs.rapids.ai/api/cuml/stable/  -> cuml.ensemble.RandomForestClassifier
                model_params = {
                    'n_estimators' : 10,                # default = 10,           :: number of trees in the forest
                    'max_depth' : 10,                   # default = 16,           :: maximum tree depth
                    'n_bins' : 14,                       # default = 9,            :: number of bins used by the split algorithm
                    'max_features': 1.0,                # default = 1.0,          :: ratio of the number of features to consider per node split
                    'seed' : 0,                         # default = None          :: seed for the random number generator, unseeded by default                
                }
            # TODO model params CPU 

            hyperparameters = {}
            try:
                # update cross-validation folds
                model_params.update( {'CV_folds': CV_folds})

                with open( self.CSP_paths['hyperparams'], 'r') as file_handle:                
                    hyperparameters = json.load(file_handle)
                    for key, value in hyperparameters.items():
                        model_params[key] = value

                    pprint.pprint( model_params )
                    return model_params

            except Exception as error:            
                self.log_to_file( str(error) )
                return {}                     
                
    def load_data( self, filename = 'dataset.orc', col_labels = None, y_label = 'ArrDelayBinary'):

        target_filename = filename

        self.log_to_file( f'\n> loading dataset from {target_filename}...\n')

        with PerfTimer() as ingestion_timer:
            if 'CPU' in self.compute_type:            
                if 'ORC' in self.data_type:
                    with open( target_filename, mode='rb') as file:
                        dataset = pyarrow_orc.ORCFile(file).read().to_pandas()
                elif 'CSV' in self.data_type:
                    dataset = pd.read_csv( target_filename, names = col_labels )
            elif 'GPU' in self.compute_type:
                if 'ORC' in self.data_type:
                    dataset = cudf.read_orc( target_filename )
                elif 'CSV' in self.data_type:
                    dataset = cudf.read_csv( target_filename, names = col_labels )
                elif 'Parquet' in self.data_type:
                    dataset = cudf.read_parquet(target_filename)

        for col in dataset.columns:
            dataset[col] = dataset[col].astype(np.float32)  # needed for random forest
        
        # Adding y_label column if it is not present
        if y_label not in dataset.columns:
            dataset[y_label] = 1.0 * (
                    dataset["ArrDelay"] > 10
                )
            dataset[y_label] = dataset[y_label].astype(np.int32)
        
        self.log_to_file( f'ingestion completed in {ingestion_timer.duration}')        
        self.log_to_file(f'dataset descriptors: {dataset.shape}\n {dataset.dtypes}\n {dataset.columns}\n')        
        return dataset, col_labels, y_label, ingestion_timer.duration
   
    def split_data ( self, dataset, y_label, train_size = .8, random_state = 0, shuffle = True  ) :
        """
        split dataset into train and test subset 
        NOTE: assumes the first column of the dataset is the classification labels
            ! in the case of sklearn, we manually filter this column in the split call
            ! in the case of cuml, the filtering happens internally 
        """
        self.log_to_file('\tsplitting train and test data')
        start_time = time.perf_counter()
        
        with PerfTimer() as split_timer:
            if 'CPU' in self.compute_type:
                X_train, X_test, y_train, y_test = sklearn_train_test_split( dataset.loc[:, dataset.columns != y_label], dataset[y_label], train_size = train_size, 
                                                                             shuffle = shuffle, random_state = random_state )
            elif 'GPU' in self.compute_type:
                X_train, X_test, y_train, y_test = cuml_train_test_split( X = dataset, y = y_label, train_size = train_size, 
                                                                          shuffle = shuffle, random_state = random_state )        
        self.log_to_file( f'\t> split completed in {split_timer.duration}')
        return X_train, X_test, y_train, y_test, split_timer.duration

    def train_model ( self, X_train, y_train, model_params ):
        self.log_to_file(f'\training {self.model_type} estimator w/ hyper-params') 
        training_time = 0       
        try:            
            if self.model_type == 'XGBoost':
                trained_model, training_time = self.fit_xgboost ( X_train, y_train, model_params )
            elif self.model_type == 'RandomForest':
                print("HERE")
                trained_model, training_time = self.fit_random_forest ( X_train, y_train, model_params )
        except Exception as error:
            self.log_to_file( '!error during model training: ' + str(error) )
        
        self.log_to_file( f'\t> finished training in {training_time:.4f} s' )
        return trained_model, training_time

    # train dlmc.xgboost model
    def fit_xgboost ( self, X_train, y_train, model_params ):        
        with PerfTimer() as train_timer:            
            train_DMatrix = xgboost.DMatrix( data = X_train, label = y_train )
            trained_model = xgboost.train(  dtrain = train_DMatrix,
                                            params = model_params,
                                            num_boost_round = model_params['num_boost_round'],
                                            verbose_eval = self.verbose_estimator )
        return trained_model, train_timer.duration
    
    # fit_xgboost_multi_GPU ()    
    # fit_random_forest_multi_GPU ()

    # train cuml.random-forest model
    def fit_random_forest ( self, X_train, y_train, model_params ):        
        if 'CPU' in self.compute_type:
            rf_model = sklearn.ensemble.RandomForestClassifier( n_estimators = model_params['n_estimators'],
                                                                max_depth = model_params['max_depth'],                                                                
                                                                max_features = model_params['max_features'], 
                                                                n_jobs = int(self.n_workers),
                                                                verbose = self.verbose_estimator )
        elif 'GPU' in self.compute_type:
            rf_model = cuml.ensemble.RandomForestClassifier ( n_estimators = model_params['n_estimators'],
                                                                max_depth = model_params['max_depth'],
                                                                n_bins = model_params['n_bins'],
                                                                max_features = model_params['max_features'],
                                                                verbose = self.verbose_estimator )        
        with PerfTimer() as train_timer:            
            trained_model = rf_model.fit( X_train, y_train)
        
        return trained_model, train_timer.duration
    
    def evaluate_test_perf ( self, trained_model, X_test, y_test ):        
        self.log_to_file(f'\tinferencing on test set')            
        with PerfTimer() as inference_timer:
            try:
                if self.model_type == 'XGBoost':
                    test_DMatrix = xgboost.DMatrix( data = X_test, label = y_test )    
                    test_accuracy = 1 - float( trained_model.eval( test_DMatrix ).split(':')[1] )

                elif self.model_type == 'RandomForest':
                    # y_test = cudf.DataFrame({'label': y_test.astype('int32') })
                    test_accuracy = trained_model.score( X_test, y_test.astype('int32') )

            except Exception as error:
                self.log_to_file( '!error during inference: ' + str(error) )

        
        self.log_to_file(f'\t> finished inference in {inference_timer.duration:.4f} s' )
        self.log_to_file(f'\n\ttest-accuracy: {test_accuracy};\n')
        return test_accuracy, inference_timer.duration
        
    # TODO: FIL inference [ ? ]
    # evaluate_perf_FIL(self, trained_model, X_test, y_test ):
    
    # TODO: global_best_model.save()
    def save_best_model ( self, global_best_model = None ):
        pass
    
    # ------------------------------------------------------
    # end of data science logic
    # ------------------------------------------------------

    def parse_compute( self, n_workers = None ):
        if 'CPU' in self.compute_type or 'GPU' in self.compute_type: 
            available_devices = self.query_compute()            
            if n_workers == -1: 
                n_workers = available_devices
            assert( n_workers <= available_devices )
            self.log_to_file (f'compute type: {self.compute_type}, n_workers: { n_workers}')            
        else: 
            raise Exception('unsupported compute type')        
        return n_workers

    def query_compute ( self ):
        available_devices = None
        if 'CPU' in self.compute_type:            
            available_devices = os.cpu_count()
            self.log_to_file( f'detected {available_devices} CPUs' )
        elif 'GPU' in self.compute_type:            
            available_devices = cupy.cuda.runtime.getDeviceCount()
            self.log_to_file( f'detected {available_devices} GPUs' )
        return available_devices

    # TODO: enumerate all visible GPUs [ ? ]
    def query_memory ( self ):        
        def print_device_memory( memory, device_ID = -1 ):
            memory_free_GB = np.array(memory.free) / np.array(10e8)
            memory_used_GB = np.array(memory.used) / np.array(10e8)
            memory_total_GB = np.array(memory.total) / np.array(10e8)
            if device_ID != -1: 
                self.log_to_file(f'device ID = {device_ID}')
            self.log_to_file(f'memory free, used, total: {memory_free_GB}, {memory_used_GB}, {memory_total_GB}')   
               
        if 'CPU' in self.compute_type:
            print_device_memory ( psutil.virtual_memory() )
            
        elif 'GPU' in self.compute_type:
            pynvml.nvmlInit()
            for iGPU in range(self.n_workers):
                handle = pynvml.nvmlDeviceGetHandleByIndex( iGPU ) 
                print_device_memory( pynvml.nvmlDeviceGetMemoryInfo( handle ) )

    def set_up_logging( self ):        
        logging_path = self.CSP_paths['output'] + '/log.txt'
        logging.basicConfig( filename= logging_path,
                             level=logging.INFO)

    def log_to_file ( self, text ):
        logging.info( text )
        print( text )
    
    def environment_check ( self ):
        self.check_dirs()    

        if self.cloud_type == 'AWS':
            try: 
                self.list_files( '/opt/ml' )
                self.log_to_file( os.environ['SM_NUM_GPUS'] )
                self.log_to_file( os.environ['SM_TRAINING_ENV'] )
                self.log_to_file( os.environ['SM_CHANNEL_TRAIN'] )                
                self.log_to_file( os.environ['SM_HPS'] )
            except: pass
        else:
            pass

    def check_dirs ( self ):
        self.log_to_file('\n> checking for AzureML paths...\n')
        
        directories_to_check = self.CSP_paths
        for iDir, val in directories_to_check.items():
            self.log_to_file(f'{val}, exists : {os.path.exists(val)}' )
        
        self.log_to_file( f'working directory = {os.getcwd()}' )

    def list_files( self, startpath ):
        print(f'\n> listing contents of {startpath}\n')
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = ' ' * 4 * (level)
            print('{}{}/'.format(indent, os.path.basename(root)))
            subindent = ' ' * 4 * (level + 1)
            for f in files:
                print('{}{}'.format(subindent, f))
           

# perf_counter = highest available timer resolution 
class PerfTimer:
    def __init__(self):
        self.start = None        
        self.duration = None        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *args):        
        self.duration = time.perf_counter() - self.start


'''
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.fit

n_estimators=100,
criterion='gini',
max_depth=None,
min_samples_split=2,
min_samples_leaf=1,
min_weight_fraction_leaf=0.0,
max_features='auto',
max_leaf_nodes=None,
min_impurity_decrease=0.0,
min_impurity_split=None,
bootstrap=True,
oob_score=False,
n_jobs=None,
random_state=None,
verbose=0,
warm_start=False,
class_weight=None,
ccp_alpha=0.0,
max_samples=None

'''