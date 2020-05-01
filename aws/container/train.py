''' 
    bring your own container logic running in each estimator
'''
import sys, os, time
import boto3, json
import numpy as np
import rapids_csp
import argparse
import pprint


DEFAULT_CONFIG = {
    'model_type'       : 'RandomForest', # RandomForest'
    'compute'          : 'GPU',
    'dataset'          : 'airline',
    'n_samples'        : 1000000,
    'dataset_type'     : 'orc',
    'dataset_filename' : 'airline_1000000.orc',
    'CV_folds'         : 2
}

def train( model_params, config_params ):

    rcml = rapids_csp.RapidsCloudML( cloud_type = 'AWS',
                                     model_type = config_params['model_type'],
                                     compute_type = config_params['compute'] )

    # environment check
    rcml.environment_check () 

    # ingest data [ post pre-processing ]
    dataset, col_labels, y_label, ingest_time = rcml.load_data ( filename = config_params['dataset_filename'] )
    rcml.query_memory()

    # classification objective requires int32 label for cuml random forest 
    dataset[y_label] = dataset[y_label].astype('int32')
    
    # ----------------------------------------------------------------------------------------------------
    # cross-validation folds 
    # ----------------------------------------------------------------------------------------------------
    accuracy_per_fold = []; train_time_per_fold = []; infer_time_per_fold = []; split_time_per_fold = []
    global_best_model = None; global_best_test_accuracy = 0

    # optional cross-validation w/ model_params['n_train_folds'] > 1
    for i_train_fold in range( config_params['CV_folds'] ):

        rcml.log_to_file( f"\nCV fold { i_train_fold } of { config_params['CV_folds'] }\n" )

        # split data
        X_train, X_test, y_train, y_test, split_time = rcml.split_data( dataset, y_label, 
                                                                        random_state = i_train_fold, 
                                                                        shuffle = True )
        split_time_per_fold += [ round( split_time, 4)]        

        ''' pre-packaged '''
        # train model            
        trained_model, training_time = rcml.train_model ( X_train, y_train, model_params )
        train_time_per_fold += [ round( training_time, 4) ]

        # evaluate perf
        test_accuracy, infer_time = rcml.evaluate_test_perf ( trained_model, X_test, y_test )
        accuracy_per_fold += [ round( test_accuracy, 4) ]
        infer_time_per_fold += [ round( infer_time, 4) ]

        # custom estimator [ TODO: instructions -- custom_entrypoint_estimator ]
        '''
        rf_model = cuml.ensemble.RandomForestClassifier ( n_estimators = model_params['n_estimators'],
                                                            max_depth = model_params['max_depth'],
                                                            n_bins = model_params['n_bins'],
                                                            max_features = model_params['max_features'],
                                                            verbose = self.verbose_estimator )
        trained_model = rf_model.fit( X_train, y_train)
        test_accuracy = trained_model.score( X_test, y_test )
        '''

        # update best model [ assumes maximization of perf metric ]
        if test_accuracy > global_best_test_accuracy :
            global_best_test_accuracy = test_accuracy
            global_best_model = trained_model

    rcml.log_to_file( f'\n best accuracy        : {np.max(accuracy_per_fold)} \n' )
    rcml.log_to_file( f'\n accuracy per fold    : {accuracy_per_fold} \n' )
    rcml.log_to_file( f'\n train-time per fold  : {train_time_per_fold} \n' )
    rcml.log_to_file( f'\n infer-time per fold  : {infer_time_per_fold} \n' )
    rcml.log_to_file( f'\n split-time per fold  : {split_time_per_fold} \n' )
    
    rcml.log_to_file( f'\n average accuracy     : {np.mean(accuracy_per_fold):.4f} \n' )
    rcml.log_to_file( f'\n average train-time   : {np.mean(train_time_per_fold):.4f} \n' )
    rcml.log_to_file( f'\n average infer-time   : {np.mean(infer_time_per_fold):.4f} \n' )
    rcml.log_to_file( f'\n average split-time   : {np.mean(split_time_per_fold):.4f} \n' )
   
    rcml.log_to_file( f'\n ingest time [once]   : {ingest_time:.4f} \n')

    # save model
    rcml.save_best_model( global_best_model )

    # ----------------------------------------------------------------------------------------------------
    # TODO: baseline [ single-node - multi-GPU w/ dask ]
    # ----------------------------------------------------------------------------------------------------    
    return 0

# use job name to define model type, compute, data
def parse_config_parameters():
    print('\nparsing job config from job filename...\n')
    config = DEFAULT_CONFIG
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
            if 'gpu' in job_name.split('-')[1].lower():
                config['compute'] = 'GPU'
            else:
                config['compute'] = 'CPU'

            # parse dataset and n_samples
            #dataset_str = job_name.split('-')[4]
            #if 'air' in dataset_str:
            config['dataset'] = 'airline' 
            
            config['n_samples'] = int( job_name.split('-')[4] )
            config['dataset_filename'] = f"{config['dataset']}_{config['n_samples']}.{config['dataset_type']}"

            # parse CV folds
            config['CV_folds'] = int(job_name.split('-')[3])

            assert(config['CV_folds'] > 0 and config['n_samples'] > 0)

        else:
            print( 'unable to parse config from job filename, loading defaults...\n')        

    except Exception as error:
        print( error )

    pprint.pprint( config, indent = 5 ); print( '\n' )
    return config
    
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
            'lambda': args.lambda_,
            'alpha': args.alpha,
            'random_state' : 0,
        }        
        if 'GPU' in config_params['compute']:
            model_params.update( { 'tree_method': 'gpu_hist' })
        else:
            model_params.update( { 'tree_method': 'hist' })

    elif 'RandomForest' in config_params['model_type']:

        # random-forest - supported hyperparameters
        parser.add_argument( '--n_estimators', type = int,   default = 200,  
                             help = 'number of trees in a random forest' )
        parser.add_argument( '--max_depth',    type = int,   default = 18,  
                             help = 'maximum tree depth' )
        parser.add_argument( '--max_features', type = float, default = .5, 
                             help = 'ratio of the number of features to consider per node split' )
        parser.add_argument( '--n_bins',       type = int,   default = 32,   
                             help = 'number of bins used by the histogram split algorithm' )

        args, unknown_args = parser.parse_known_args( input_args )

        # random forest
        model_params = {            
            'max_depth' : args.max_depth,
            'n_estimators' : args.n_estimators,        
            'max_features': args.max_features,
            'n_bins' : args.n_bins,
            'seed' : 0,        
        }
    else:
        raise Exception(f"!error: unknown model type {config_params['model_type']}")

    print(f'known arguments {args}')
    print(f'unknown arguments {unknown_args}')
    print(f'printing parsed model parameters: {model_params}')
    return model_params

if __name__ == "__main__":
    config_params = parse_config_parameters ()
    model_params = parse_model_parameters ( sys.argv[1:], config_params )    
    train( model_params, config_params )
    sys.exit(0)