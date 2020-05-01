import sys, os, time, logging
import warnings; warnings.simplefilter("ignore", UserWarning)

# i/o
import logging, json, pprint

default_sagemaker_paths = {
    'base' : '/opt/ml',
    'code' : '/opt/ml/code',
    'data' : '/opt/ml/input',
    'train_data' : '/opt/ml/input/data/training',
    'hyperparams' : '/opt/ml/input/config/hyperparameters.json',
    'model' : '/opt/ml/model',
    'output' : '/opt/ml/output',
}

# TODO multi-cpu baseline [ on sagemaker notebook ? ]
# TODO xgboost CPU

class RapidsCloudML ( object ):

    def __init__ ( self, cloud_type = 'AWS', 
                   model_type = 'XGBoost', 
                   data_type = 'ORC', 
                   compute_type = 'single-GPU', 
                   n_workers = -1, 
                   verbose_estimator = False,
                   CSP_paths = default_sagemaker_paths ):

        self.CSP_paths = CSP_paths
        self.cloud_type = cloud_type        
        self.model_type = model_type
        self.data_type = data_type
        self.compute_type = compute_type
        self.verbose_estimator = verbose_estimator
        
        self.version_check()
        self.n_workers = self.parse_compute( n_workers )
        self.query_memory()

    def load_data( self, filename = 'dataset.orc', col_labels = None, y_label = 'ArrDelayBinary'):                
        
        if 'CPU' in self.compute_type:
            try: import pandas, pyarrow; from pyarrow import orc
            except Exception as error: print ( f'! CPU import error : {error}' )
        elif 'GPU' in self.compute_type:
            try: import cudf
            except Exception as error: print ( f'! GPU import error : {error}' )        

        target_filename = self.CSP_paths['train_data'] + '/' + filename
        self.log_to_file( f'\n> loading dataset from {target_filename}...\n')
                
        with PerfTimer ( self, 'ingestion_timer') as ingestion_timer:
            if 'CPU' in self.compute_type:            
                if 'ORC' in self.data_type:
                    with open( target_filename, mode='rb') as file:
                        dataset = pyarrow.orc.ORCFile(file).read().to_pandas()
                elif 'CSV' in self.data_type:
                    dataset = pandas.read_csv( target_filename, names = col_labels )
            elif 'GPU' in self.compute_type:
                if 'ORC' in self.data_type:
                    dataset = cudf.read_orc( target_filename )
                elif 'CSV' in self.data_type:
                    dataset = cudf.read_csv( target_filename, names = col_labels )
        
        # self.log_to_file(f'dataset descriptors: {dataset.shape}\n {dataset.dtypes}\n {dataset.columns}\n')        
        return dataset, col_labels, y_label, ingestion_timer.duration
   
    def split_data ( self, dataset, y_label, train_size = .8, random_state = 0, shuffle = True  ) :
        """
        split dataset into train and test subset 
        NOTE: assumes the first column of the dataset is the classification labels
            ! in the case of sklearn, we manually filter this column in the split call
            ! in the case of cuml, the filtering happens internally 
        """

        if 'CPU' in self.compute_type:
            try: 
                import pandas, sklearn
                from sklearn.model_selection import train_test_split as sklearn_train_test_split
            except Exception as error: print ( f'! CPU import error : {error}' )
        elif 'GPU' in self.compute_type:
            try: 
                import cudf
                from cuml.preprocessing.model_selection import train_test_split as cuml_train_test_split                
            except Exception as error: print ( f'! GPU import error : {error}' )        


        with PerfTimer( self, 'split_timer') as split_timer:
            if 'CPU' in self.compute_type:
                X_train, X_test, y_train, y_test = sklearn_train_test_split( dataset.loc[:, dataset.columns != y_label], dataset[y_label], train_size = train_size, 
                                                                             shuffle = shuffle, random_state = random_state )
            elif 'GPU' in self.compute_type:
                X_train, X_test, y_train, y_test = cuml_train_test_split( X = dataset, y = y_label, train_size = train_size, 
                                                                          shuffle = shuffle, random_state = random_state )                                                                                        
                                                          
        return X_train, X_test, y_train, y_test, split_timer.duration

    def train_model ( self, X_train, y_train, model_params ):
        try:            
            if self.model_type == 'XGBoost':
                trained_model, training_time = self.fit_xgboost ( X_train, y_train, model_params )
            elif self.model_type == 'RandomForest':
                trained_model, training_time = self.fit_random_forest ( X_train, y_train, model_params )
        except Exception as error:
            self.log_to_file( '!error during model training: ' + str(error) )
        
        return trained_model, training_time

    # train dlmc.xgboost model
    def fit_xgboost ( self, X_train, y_train, model_params ):
        import xgboost

        with PerfTimer(self, 'train_xgboost_timer') as train_timer:            
            train_DMatrix = xgboost.DMatrix( data = X_train, label = y_train )
            trained_model = xgboost.train( dtrain = train_DMatrix,
                                           params = model_params,
                                           num_boost_round = model_params['num_boost_round'],
                                           verbose_eval = self.verbose_estimator )

        return trained_model, train_timer.duration
    
    # fit_xgboost_multi_GPU ()    
    # fit_random_forest_multi_GPU ()

    # train cuml.random-forest model
    def fit_random_forest ( self, X_train, y_train, model_params ):        
        if 'CPU' in self.compute_type:
            try: 
                import pandas, sklearn
                from sklearn import ensemble
            except Exception as error: print ( f'! CPU import error : {error}' )

            rf_model = sklearn.ensemble.RandomForestClassifier ( n_estimators = model_params['n_estimators'],
                                                                 max_depth = model_params['max_depth'],                                                                
                                                                 max_features = model_params['max_features'], 
                                                                 n_jobs = -1, # int(self.n_workers),
                                                                 verbose = self.verbose_estimator )
        elif 'GPU' in self.compute_type:
            try: 
                import cudf, cuml
            except Exception as error: print ( f'! CPU import error : {error}' )

            rf_model = cuml.ensemble.RandomForestClassifier ( n_estimators = model_params['n_estimators'],
                                                              max_depth = model_params['max_depth'],
                                                              n_bins = model_params['n_bins'],
                                                              max_features = model_params['max_features'],
                                                              verbose = self.verbose_estimator )

        with PerfTimer(self, 'train_RandomForest_timer') as train_timer:
            trained_model = rf_model.fit( X_train, y_train)
        
        return trained_model, train_timer.duration
    
    def evaluate_test_perf ( self, trained_model, X_test, y_test ):
        if 'CPU' in self.compute_type:
            try: import pandas, sklearn, xgboost
            except Exception as error: print(error)
        elif 'GPU' in self.compute_type:
            try: import cudf, cuml, xgboost
            except Exception as error: print(error)
        with PerfTimer(self, 'score_timer') as score_timer:
            try:
                if self.model_type == 'XGBoost':
                    test_DMatrix = xgboost.DMatrix( data = X_test, label = y_test )    
                    test_score = trained_model.eval( test_DMatrix )
                    test_accuracy = 1 - float( test_score.split(':')[1] )

                elif self.model_type == 'RandomForest':
                    test_accuracy = trained_model.score( X_test, y_test, fil_sparse_format=True ) # algo='tree_reorg', 

            except Exception as error:
                self.log_to_file( '!error during inference: ' + str(error) )
        
        self.log_to_file(f'\n\ttest-accuracy: {test_accuracy};\n')
        return test_accuracy, score_timer.duration
        
    # TODO: FIL inference [ ? ]
    # evaluate_perf_FIL(self, trained_model, X_test, y_test ):
    
    # TODO: global_best_model.save()
    def save_best_model ( self, global_best_model = None ):
        pass
    
    # ------------------------------------------------------
    # end of data science logic
    # ------------------------------------------------------

    def version_check ( self ):

        if 'CPU' in self.compute_type:
            try: 
                import pandas, numpy, sklearn
                self.log_to_file( f'pandas version  : {pandas.__version__}' )
                self.log_to_file( f'sklearn version : {sklearn.__version__}' )
                self.log_to_file( f'numpy version   : {numpy.__version__}\n' )

            except Exception as error: print ( f'! CPU DS stack import error: {error}' )
        elif 'GPU' in self.compute_type:
            try: 
                import cudf, cuml, cupy, dask
                self.log_to_file( f'\ncudf version  : {cudf.__version__}' )
                self.log_to_file( f'cuml version  : {cuml.__version__}' )
                self.log_to_file( f'cupy version  : {cupy.__version__}' )
                self.log_to_file( f'dask version  : {dask.__version__}' )

            except Exception as error: print ( f'! GPU DS stack import error: {error}' )  

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
            import cupy
            available_devices = cupy.cuda.runtime.getDeviceCount()
            self.log_to_file( f'detected {available_devices} GPUs' )
        return available_devices

    def query_memory ( self ):        
        import numpy
        def print_device_memory( memory, device_ID = -1 ):
            memory_free_GB = numpy.array(memory.free) / numpy.array(10e8)
            memory_used_GB = numpy.array(memory.used) / numpy.array(10e8)
            memory_total_GB = numpy.array(memory.total) / numpy.array(10e8)
            if device_ID != -1: 
                self.log_to_file(f'device ID = {device_ID}')
            self.log_to_file(f'memory free, used, total: {memory_free_GB}, {memory_used_GB}, {memory_total_GB}')   
               
        if 'CPU' in self.compute_type:
            import psutil
            print_device_memory ( psutil.virtual_memory() )
            
        elif 'GPU' in self.compute_type:
            import pynvml
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
        self.log_to_file('\n> checking for sagemaker paths...\n')
        
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
        self.rcml.log_to_file(f"|-> {self.name_string} : {self.duration:.4f}\n")
        

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