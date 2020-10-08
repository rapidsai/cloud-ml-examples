import os
import json 
import argparse
import glob
import pprint

import HPODatasets

"""  Cloud integrated RAPIDS HPO functionality with AWS SageMaker focus """

class HPOConfig ( object ):

    sagemaker_directory_structure = {
        'train_data' : '/opt/ml/input/data/training', 
        'model_store' : '/opt/ml/model',
        'output_artifacts' : '/opt/ml/output'
    }    

    def __init__ ( self, input_args, 
                   directory_structure = sagemaker_directory_structure,
                   worker_limit = None ):
        
        # parse configuration from job-name
        self.dataset_type, self.model_type, \
            self.compute_type, self.cv_folds = self.parse_configuration_from_environment()

        # parse input parameters for HPO        
        self.model_params = self.parse_hyper_parameter_inputs ( input_args )
        
        # prase dataset files/paths and dataset columns, labels, dtype [ BYOD vs airline ]
        self.target_files, self.input_file_type, \
            self.dataset_columns, self.label_column, \
                self.dataset_dtype = self.detect_data_inputs( directory_structure )
        
        self.model_store_directory = directory_structure ['model_store']
        self.output_artifacts_directory = directory_structure ['output_artifacts']

    def parse_configuration_from_environment ( self ):
        """ 
        Parse the ENV variables [ set in the dockerfile ] to determine configuration settings 
        """    

        print('\nparsing configuration choices from environment settings...\n')

        dataset_type = 'Airline' 
        model_type = 'RandomForest'
        compute_type = 'single-GPU'
        cv_folds = 3

        try:
            # print ( os.environ ); print ( '\n')
            
            # parse dataset choice
            dataset_selection = os.environ['DATASET_DIRECTORY'].lower()
            if dataset_selection in [ '1_year', '3_year', '10_year']:
                dataset_type = 'Airline'
            elif dataset_selection in ['nyc_taxi']:
                dataset_type = 'NYCTaxi'
            else:
                dataset_type = 'BYOData'

            # parse model type
            model_selection = os.environ['ALGORITHM_CHOICE'].lower()
            if model_selection in ['randomforest']:
                model_type = 'RandomForest'
            elif model_selection in ['xgboost']:
                model_type = 'XGBoost'

            # parse compute choice
            compute_selection = os.environ['ML_WORKFLOW_CHOICE'].lower()
            if 'multigpu' in compute_selection:
                compute_type = 'multi-GPU'
            elif 'multicpu' in compute_selection:
                compute_type = 'multi-CPU'
            elif 'singlecpu' in compute_selection:
                compute_type = 'single-CPU'
            elif 'singlegpu' in compute_selection:
                compute_type = 'single-GPU'                
                
            # parse CV folds
            cv_folds = int( os.environ['CV_FOLDS'] )

        except Exception as error:
            print( f'{error} ! unable to parse job name, loading defaults' )
                
        assert ( dataset_type in ['Airline', 'NYCTaxi','BYOData'] )
        assert ( model_type   in ['RandomForest', 'XGBoost'] )
        assert ( compute_type in ['single-GPU', 'multi-GPU', 'single-CPU', 'multi-CPU'] )
        assert ( cv_folds >= 1 )
        
        print(f'  Dataset: {dataset_type}\n'
              f'  Compute: {compute_type}\n'
              f'  Algorithm: {model_type}\n'
              f'  CV_folds: {cv_folds}\n')

        return dataset_type, model_type, compute_type, cv_folds

    def parse_hyper_parameter_inputs ( self, input_args ):
        """ Parse hyperparmeters that are fed in by the HPO orchestrator [e.g., SageMaker ]. """
        print('parsing model hyperparameters from command line arguments...\n')
        parser = argparse.ArgumentParser ()

        if 'XGBoost' in self.model_type:
            parser.add_argument( '--max_depth',       type = int,   default = 5 )
            parser.add_argument( '--num_boost_round', type = int,   default = 10 )            
            parser.add_argument( '--subsample',       type = float, default = .9 )
            parser.add_argument( '--learning_rate',   type = float, default = 0.3 )            
            parser.add_argument( '--reg_lambda',      type = float, default = 1 )            
            parser.add_argument( '--gamma',           type = float, default = 0. )            
            parser.add_argument( '--alpha',           type = float, default = 0. )
            parser.add_argument( '--seed',            type = int,   default = 0 )
            
            args, unknown_args = parser.parse_known_args( input_args )
            
            model_params = {            
                'max_depth' : args.max_depth,
                'num_boost_round': args.num_boost_round,
                'learning_rate': args.learning_rate,
                'gamma': args.gamma,
                'lambda': args.reg_lambda,
                'random_state' : args.seed,
                'verbosity' : 0,
                'seed': args.seed,   
                'objective' : 'binary:logistic'
            }        

            if 'single-CPU' in self.compute_type:
                model_params.update( { 'nthreads': os.cpu_count() })

            if 'GPU' in self.compute_type:
                model_params.update( { 'tree_method': 'gpu_hist' })
            else:
                model_params.update( { 'tree_method': 'hist' })
            
        elif 'RandomForest' in self.model_type:
            parser.add_argument( '--max_depth'   , type = int,   default = 15 )
            parser.add_argument( '--n_estimators', type = int,   default = 100 )            
            parser.add_argument( '--max_features', type = float, default = 1.0 )
            parser.add_argument( '--n_bins'      , type = float, default = 64 )
            parser.add_argument( '--bootstrap'   , type = bool,  default = True )
            parser.add_argument( '--random_state', type = int,   default = 0 )

            args, unknown_args = parser.parse_known_args( input_args )

            model_params = {            
                'max_depth' : args.max_depth,
                'n_estimators' : args.n_estimators,        
                'max_features': args.max_features,
                'n_bins' : args.n_bins,
                'bootstrap' : args.bootstrap,
                'random_state' : args.random_state
            }
            
        else:
            raise Exception(f"!error: unknown model type {self.model_type}")

        pprint.pprint( model_params, indent = 5 ); print( '\n' )
        return model_params

    def detect_data_inputs ( self, directory_structure ):
        """ 
        Scan dataset to determine which files to ingest and modify path based on compute_type.
        This should help confirm that a correct AWS S3 bucket choice has been made.
        Notes: single-CPU pandas read_parquet needs a directory input
               single-GPU cudf read_parquet needs a list of files
               multi-CPU/GPU can accept either a list or a directory
        """
        parquet_files = glob.glob( directory_structure['train_data'] + '/*.parquet' )
        csv_files = glob.glob( directory_structure['train_data'] + '/*.csv' )

        if len( csv_files ):
            print('CSV input files detected')
            target_files = csv_files
            input_file_type = 'CSV'

        elif len( parquet_files ):
            print('Parquet input files detected')
            if 'single-CPU' in self.compute_type:
                # pandas read_parquet needs a directory input
                target_files = directory_structure['train_data'] + '/'
            else:                
                target_files = parquet_files
            input_file_type = 'Parquet'
        else:
            raise Exception ( f"! No [CSV or Parquet] input files detected")
        
        n_datafiles = len( target_files )
        assert( n_datafiles > 0 )

        pprint.pprint( target_files ); print('\n')
        print( f'detected {n_datafiles} files as input \n')

        if 'Airline' in self.dataset_type:
            print( ' using Airline dataset ')
            dataset_columns = HPODatasets.airline_feature_columns
            dataset_label_column = HPODatasets.airline_label_column
            dataset_dtype = HPODatasets.airline_dtype
            
        elif 'NYCTaxi' in self.dataset_type:
            print( ' using NYCTaxi dataset ')
            dataset_columns = HPODatasets.nyctaxi_feature_columns
            dataset_label_column = HPODatasets.nyctaxi_label_column
            dataset_dtype = HPODatasets.nyctaxi_dtype
            
        elif 'BYOData' in self.dataset_type:
            print( ' using BYOD dataset ')
            dataset_columns = HPODatasets.BYOD_feature_columns
            dataset_label_column = HPODatasets.BYOD_label_column
            dataset_dtype = HPODatasets.BYOD_dtype

        return target_files, input_file_type, \
               dataset_columns, dataset_label_column, dataset_dtype