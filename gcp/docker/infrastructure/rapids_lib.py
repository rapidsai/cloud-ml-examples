# os
import sys, os, time, logging

# CPU DS stack
import pandas as pd
import numpy as np
import sklearn

# GPU DS stack [ rapids ]
import gcsfs

# scaling library
import dask

# data ingestion [ CPU ]
from pyarrow import orc as pyarrow_orc

# ML models
from sklearn import ensemble
import xgboost

# data set splits
from sklearn.model_selection import train_test_split as sklearn_train_test_split

# device query
##hack
try:
    import cudf, cuml
    from cuml.preprocessing.model_selection import train_test_split as cuml_train_test_split
    import pynvml
    import cupy
except:
    print("Caught import failures -- probably missing GPU")

# memory query
import psutil

# i/o
import logging, json, pprint

default_sagemaker_paths = {
    'base': '/opt/ml',
    'code': '/opt/ml/code',
    'data': '/opt/ml/input',
    'train_data': '/opt/ml/input/data/training',
    'hyperparams': '/opt/ml/input/config/hyperparameters.json',
    'model': '/opt/ml/model',
    'output': '/opt/ml/output',
}


class RapidsCloudML(object):

    def __init__(self, cloud_type='AWS',
                 model_type='XGBoost',
                 data_type='ORC',
                 compute_type='single-GPU',
                 n_workers=-1,
                 verbose_estimator=False,
                 CSP_paths=default_sagemaker_paths):

        self.CSP_paths = CSP_paths
        self.cloud_type = cloud_type
        self.model_type = model_type
        self.data_type = data_type
        self.compute_type = compute_type
        self.verbose_estimator = verbose_estimator
        self.n_workers = self.parse_compute(n_workers)
        self.query_memory()

    def _read_orc(self, filename):
        if ('CPU' in self.compute_type):
            if (filename.startswith('gs://')):
                fs = gcsfs.GCSFileSystem()
                with fs.open(filename, mode='rb') as file:
                    dataset = pyarrow_orc.ORCFile(file).read().to_pandas()
            else:
                with open(filename, mode='rb') as file:
                    dataset = pyarrow_orc.ORCFile(file).read().to_pandas()

        elif ('GPU' in self.compute_type):
            dataset = cudf.read_orc(filename)

        return dataset

    def _read_csv(self, filename, col_labels):
        if ('CPU' in self.compute_type):
            dataset = pd.read_csv(filename, names=col_labels)
        elif ('GPU' in self.compute_type):
            dataset = cudf.read_csv(filename, names=col_labels)

        return dataset

    def load_data(self, filename='dataset.orc', col_labels=None, y_label='ArrDelayBinary'):
        target_filename = self.CSP_paths['train_data'] + '/' + filename
        self.log_to_file(f'\n> loading dataset from {target_filename}...\n')

        with PerfTimer() as ingestion_timer:
            if 'ORC' in self.data_type:
                dataset = self._read_orc(target_filename)
            elif 'CSV' in self.data_type:
                dataset = self._read_csv(target_filename, names=col_labels)

        self.log_to_file(f'ingestion completed in {ingestion_timer.duration}')
        self.log_to_file(f'dataset descriptors: {dataset.shape}\n {dataset.dtypes}\n {dataset.columns}\n')

        return dataset, col_labels, y_label, ingestion_timer.duration

    def split_data(self, dataset, y_label, train_size=.8, random_state=0, shuffle=True):
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
                X_train, X_test, y_train, y_test = sklearn_train_test_split(dataset.loc[:, dataset.columns != y_label],
                                                                            dataset[y_label], train_size=train_size,
                                                                            shuffle=shuffle, random_state=random_state)
            elif 'GPU' in self.compute_type:
                X_train, X_test, y_train, y_test = cuml_train_test_split(X=dataset, y=y_label, train_size=train_size,
                                                                         shuffle=shuffle, random_state=random_state)


        self.log_to_file(f'\t> split completed in {split_timer.duration}')
        return X_train, X_test, y_train, y_test, split_timer.duration

    def train_model(self, X_train, y_train, model_params):
        self.log_to_file(f'\ttraining {self.model_type} estimator w/ hyper-params')
        pprint.pprint(model_params, indent=10)
        print(f"model type: {self.model_type}\n compute type: {self.compute_type}\n dataset dtype: {type(X_train)}")

        try:
            if self.model_type == 'XGBoost':
                trained_model, training_time = self.fit_xgboost(X_train, y_train, model_params)
            elif self.model_type == 'RandomForest':
                trained_model, training_time = self.fit_random_forest(X_train, y_train, model_params)

        except Exception as error:
            self.log_to_file('!error during model training: ' + str(error))
            raise

        self.log_to_file(f'\t> finished training in {training_time:.4f} s')
        return trained_model, training_time

    # train dlmc.xgboost model
    def fit_xgboost(self, X_train, y_train, model_params):
        with PerfTimer() as train_timer:
            train_DMatrix = xgboost.DMatrix(data=X_train, label=y_train)
            trained_model = xgboost.train(dtrain=train_DMatrix,
                                          params=model_params,
                                          num_boost_round=model_params['num_boost_round'],
                                          verbose_eval=self.verbose_estimator)
        return trained_model, train_timer.duration

    # fit_xgboost_multi_GPU ()
    # fit_random_forest_multi_GPU ()

    # train cuml.random-forest model
    def fit_random_forest(self, X_train, y_train, model_params):
        if 'CPU' in self.compute_type:
            rf_model = sklearn.ensemble.RandomForestClassifier(n_estimators=model_params['n_estimators'],
                                                               max_depth=model_params['max_depth'],
                                                               max_features=model_params['max_features'],
                                                               n_jobs=int(self.n_workers),
                                                               verbose=self.verbose_estimator)
        elif 'GPU' in self.compute_type:
            rf_model = cuml.ensemble.RandomForestClassifier(n_estimators=model_params['n_estimators'],
                                                            max_depth=model_params['max_depth'],
                                                            n_bins=model_params['n_bins'],
                                                            max_features=model_params['max_features'],
                                                            verbose=self.verbose_estimator)
        with PerfTimer() as train_timer:
            trained_model = rf_model.fit(X_train, y_train)

        return trained_model, train_timer.duration

    def evaluate_test_perf(self, trained_model, X_test, y_test):
        self.log_to_file(f'\tinferencing on test set')
        with PerfTimer() as inference_timer:
            try:
                if self.model_type == 'XGBoost':
                    test_DMatrix = xgboost.DMatrix(data=X_test, label=y_test)
                    test_accuracy = 1 - float(trained_model.eval(test_DMatrix).split(':')[1])

                elif self.model_type == 'RandomForest':
                    # y_test = cudf.DataFrame({'label': y_test.astype('int32') })
                    test_accuracy = trained_model.score(X_test, y_test.astype('int32'))

            except Exception as error:
                self.log_to_file('!error during inference: ' + str(error))
                raise

        self.log_to_file(f'\t> finished inference in {inference_timer.duration:.4f} s')
        return test_accuracy, inference_timer.duration

    # TODO: FIL inference [ ? ]
    # evaluate_perf_FIL(self, trained_model, X_test, y_test ):

    # TODO: global_best_model.save()
    def save_best_model(self, global_best_model=None):
        pass

    # ------------------------------------------------------
    # end of data science logic
    # ------------------------------------------------------

    def parse_compute(self, n_workers=None):
        if 'CPU' in self.compute_type or 'GPU' in self.compute_type:
            available_devices = self.query_compute()
            if n_workers == -1:
                n_workers = available_devices
            assert (n_workers <= available_devices)
            self.log_to_file(f'compute type: {self.compute_type}, n_workers: {n_workers}')
        else:
            raise Exception('unsupported compute type')
        return n_workers

    def query_compute(self):
        available_devices = None
        if 'CPU' in self.compute_type:
            available_devices = os.cpu_count()
            self.log_to_file(f'detected {available_devices} CPUs')
        elif 'GPU' in self.compute_type:
            available_devices = cupy.cuda.runtime.getDeviceCount()
            self.log_to_file(f'detected {available_devices} GPUs')
        return available_devices

    # TODO: enumerate all visible GPUs [ ? ]
    def query_memory(self):
        def print_device_memory(memory, device_ID=-1):
            memory_free_GB = np.array(memory.free) / np.array(10e8)
            memory_used_GB = np.array(memory.used) / np.array(10e8)
            memory_total_GB = np.array(memory.total) / np.array(10e8)
            if device_ID != -1:
                self.log_to_file(f'device ID = {device_ID}')
            self.log_to_file(f'memory free, used, total: {memory_free_GB}, {memory_used_GB}, {memory_total_GB}')

        if 'CPU' in self.compute_type:
            print_device_memory(psutil.virtual_memory())

        elif 'GPU' in self.compute_type:
            pynvml.nvmlInit()
            for iGPU in range(self.n_workers):
                handle = pynvml.nvmlDeviceGetHandleByIndex(iGPU)
                print_device_memory(pynvml.nvmlDeviceGetMemoryInfo(handle))

    def set_up_logging(self):
        logging_path = self.CSP_paths['output'] + '/log.txt'
        logging.basicConfig(filename=logging_path,
                            level=logging.INFO)

    def log_to_file(self, text):
        logging.info(text)
        print(text)

    def environment_check(self):
        self.check_dirs()

        if self.cloud_type == 'AWS':
            try:
                self.list_files('/opt/ml')
                self.log_to_file(os.environ['SM_NUM_GPUS'])
                self.log_to_file(os.environ['SM_TRAINING_ENV'])
                self.log_to_file(os.environ['SM_CHANNEL_TRAIN'])
                self.log_to_file(os.environ['SM_HPS'])
            except:
                pass
        else:
            pass

    def check_dirs(self):
        self.log_to_file('\n> checking for sagemaker paths...\n')

        directories_to_check = self.CSP_paths
        for iDir, val in directories_to_check.items():
            self.log_to_file(f'{val}, exists : {os.path.exists(val)}')

        self.log_to_file(f'working directory = {os.getcwd()}')

    def list_files(self, startpath):
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
