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


import time
import os

import cudf 
import xgboost
import joblib

from cuml.preprocessing.model_selection import train_test_split
from cuml.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score

from MLWorkflow import MLWorkflow, timer_decorator

class MLWorkflowSingleGPU ( MLWorkflow ):
    """ Single-GPU Workflow """

    def __init__(self, hpo_config ):
        print( 'Single-GPU Workflow \n')
        self.start_time = time.perf_counter()

        self.hpo_config = hpo_config
        self.dataset_cache = None
        
        self.cv_fold_scores = []
        self.best_score = -1

    @timer_decorator
    def ingest_data ( self ): 
        """ Ingest dataset, CSV and Parquet supported """

        if self.dataset_cache is not None:
            print( 'skipping ingestion, using cache')
            return self.dataset_cache

        if 'Parquet' in self.hpo_config.input_file_type:
            dataset = cudf.read_parquet( self.hpo_config.target_files,
                                         columns = self.hpo_config.dataset_columns )

        elif 'CSV' in self.hpo_config.input_file_type:
            if isinstance( self.hpo_config.target_files, list):
                filepath = self.hpo_config.target_files[0]
            elif isinstance( self.hpo_config.target_files, str):
                filepath = self.hpo_config.target_files

            print( self.hpo_config.dataset_columns )
            dataset = cudf.read_csv( filepath,
                                     names = self.hpo_config.dataset_columns, 
                                     header = 0 )
        
        print( f'ingested {self.hpo_config.input_file_type} dataset; shape = {dataset.shape}' )
        self.dataset_cache = dataset
        return dataset

    @timer_decorator
    def handle_missing_data ( self, dataset ): 
        """ Drop samples with missing data [ inplace i.e., do not copy dataset ] """
        dataset = dataset.dropna()
        return dataset

    @timer_decorator
    def split_dataset ( self, dataset, random_state ): 
        """ Split into train and test data subsets, using CV-fold index for randomness """
        
        label_column = self.hpo_config.label_column
        
        X_train, X_test, y_train, y_test = train_test_split( dataset, label_column, random_state = random_state )

        return ( X_train.astype( self.hpo_config.dataset_dtype ),
                 X_test.astype( self.hpo_config.dataset_dtype ),
                 y_train.astype( self.hpo_config.dataset_dtype ),
                 y_test.astype( self.hpo_config.dataset_dtype ) )

    @timer_decorator
    def fit ( self, X_train, y_train ):       
        """ Fit decision tree model [ architecture defined by HPO parameters ] """
        if 'XGBoost' in self.hpo_config.model_type:
            print('fit xgboost model')
            dtrain = xgboost.DMatrix(data = X_train, label = y_train)
            trained_model = xgboost.train( dtrain = dtrain, params = self.hpo_config.model_params,
                                           num_boost_round = self.hpo_config.model_params['num_boost_round'] )

        elif 'RandomForest' in self.hpo_config.model_type:
            print('fit randomforest model')
            trained_model = RandomForestClassifier ( n_estimators = self.hpo_config.model_params['n_estimators'],
                                                     max_depth = self.hpo_config.model_params['max_depth'],
                                                     max_features = self.hpo_config.model_params['max_features'],
                                                     n_bins = self.hpo_config.model_params['n_bins']) .fit( X_train, y_train.astype('int32') )
        return trained_model 

    @timer_decorator
    def predict ( self, trained_model, X_test, threshold = 0.5 ):
        """ Inference with the trained model on the unseen test data """

        print('predict with trained model ')
        if 'XGBoost' in self.hpo_config.model_type:
            dtest = xgboost.DMatrix( X_test )
            predictions = trained_model.predict( dtest )
            predictions = (predictions > threshold ) * 1.0
        elif 'RandomForest' in self.hpo_config.model_type:
            predictions = trained_model.predict( X_test )

        return predictions

    @timer_decorator
    def score ( self, y_test, predictions ): 
        """ Score predictions vs ground truth labels on test data """
        score = accuracy_score ( y_test.astype( self.hpo_config.dataset_dtype ),
                                predictions.astype( self.hpo_config.dataset_dtype ) )

        print(f'score = {round(score,5)}')
        self.cv_fold_scores.append( score )
        return score

    def save_best_model ( self, score, trained_model, filename = 'saved_model' ): 
        """  Persist/save model that sets a new high score """

        if score > self.best_score:
            self.best_score = score
            print('saving high-scoring model')
            output_filename = os.path.join( self.hpo_config.model_store_directory, filename )
            if 'XGBoost' in self.hpo_config.model_type:
                trained_model.save_model( f'{output_filename}_sgpu_xgb' )
            elif 'RandomForest' in self.hpo_config.model_type:
                joblib.dump ( trained_model, f'{output_filename}_sgpu_rf' )
            
    
    def cleanup ( self, i_fold ):
        print('end of cv-fold \n') 
    
    def emit_final_score ( self ):
        """ Emit score for parsing by the cloud HPO orchestrator """
        exec_time = time.perf_counter() - self.start_time
        print(f'total_time = {exec_time:.5f} s ')

        if self.hpo_config.cv_folds > 1 :
            print(f'cv-fold scores : {self.cv_fold_scores} \n')

        final_score = sum(self.cv_fold_scores) / len(self.cv_fold_scores) # average

        print(f'final-score: {final_score}; \n')