import random, uuid
import pandas
  
def new_job_name_from_config ( dataset_directory, code_choice, 
                               algorithm_choice, cv_folds,
                               instance_type, trim_limit = 32 ):
    """ Build a jobname string that captures the HPO configuration options.
        This string will be parsed by the worker containers 
        [ see parse_job_name in rapids_cloud_ml.py ].
    """
    job_name = None    
    try:
        data_choice_str = dataset_directory.split('_')[0] + 'y'        
        code_choice_str = code_choice[0] + code_choice[-3:]
        
        if 'randomforest' in algorithm_choice.lower() : algorithm_choice_str = 'RF'
        if 'xgboost' in algorithm_choice.lower() : algorithm_choice_str = 'XGB'    
        
        instance_type_str = '-'.join( instance_type.split('.')[1:] )        

        random_8char_str = ''.join( random.choices( uuid.uuid4().hex, k=8 ) )        
        
        job_name = f"{data_choice_str}-{code_choice_str}" \
                    f"-{algorithm_choice_str}-{cv_folds}cv"\
                    f"-{instance_type_str}-{random_8char_str}"
        
        job_name = job_name[:trim_limit]
        
        print ( f'generated job name : {job_name}\n')
        
    except Exception as error: 
        print( f'ERROR: unable to generate job name: {error}' )
    
    return job_name

def recommend_instance_type ( code_choice, dataset_directory  ):
    """ Based on the code and [airline] dataset-size choices we recommend instance types 
        that we've tested and are known to work. Feel free to ignore/make a different choice.
    """
    recommended_instance_type = None
    
    if 'CPU' in code_choice and dataset_directory in [ '1_year', '3_year' ]:
        detail_str =  '16 cpu cores, 64GB memory'
        recommended_instance_type = 'ml.m5.4xlarge' 

    elif 'CPU' in code_choice and dataset_directory in [ '10_year']:
        detail_str =  '96 cpu cores, 384GB memory'
        recommended_instance_type = 'ml.m5.24xlarge'

    if code_choice == 'singleGPU': 
        detail_str =  '1x V100, 16GB GPU memory, 61GB CPU memory'
        recommended_instance_type = 'ml.p3.2xlarge' 
        assert( dataset_directory not in [ '10_year'] ) # ! switch to multi-GPU

    elif code_choice == 'multiGPU':
        detail_str =  '4x V100, 64GB GPU memory,  244GB CPU memory'
        recommended_instance_type = 'ml.p3.8xlarge'
    
    print( f'recommended instance type : {recommended_instance_type} \n'\
           f'instance details          : {detail_str}' )
    
    return recommended_instance_type

def validate_dockerfile ( rapids_base_container, dockerfile_name = 'Dockerfile'):
    
    # validate that our desired rapids image matches the Dockerfile
    with open( dockerfile_name, 'r') as dockerfile_handle: 
        if rapids_base_container not in dockerfile_handle.read():
            raise Exception('Dockerfile base layer [i.e. FROM statment] does not match the variable rapids_base_container')
            
def summarize_choices( s3_data_URI, code_choice, 
                       algorithm_choice, cv_folds,
                       instance_type, use_spot_instances_flag,
                       search_strategy, max_jobs, max_parallel_jobs,
                       max_duration_of_experiment_seconds ):
    """ Print the configuration choices, often useful before submitting large jobs """
    print( f'S3 uri           =\t{s3_data_URI}')
    print( f'compute          =\t{code_choice}')
    print( f'algorithm        =\t{algorithm_choice}, {cv_folds} cv-fold')
    print( f'instance         =\t{instance_type}')
    print( f'spot instances   =\t{use_spot_instances_flag}')
    print( f'hpo strategy     =\t{search_strategy}')
    print( f'max_experiments  =\t{max_jobs}')
    print( f'max_parallel     =\t{max_parallel_jobs}')
    print( f'max runtime      =\t{max_duration_of_experiment_seconds} sec')    

def print_hpo_ranges ( hyperparameter_ranges ):
    """ Print HPO paramter search ranges """
    for key, value in hyperparameter_ranges.items():
        print( f'> range of {key}\t: {value.min_value} , {value.max_value} ')