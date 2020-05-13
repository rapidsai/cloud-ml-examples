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
    logic running in each HPO estimator
"""

import sys, numpy as np
import rapids_cloud_ml

DEFAULT_CONFIG = {
    'model_type'       : 'RandomForest', # XGBoost
    'compute'          : 'multi-CPU',
    'dataset'          : 'airline',
    'dataset_filename' : '*.parquet',
    'CV_folds'         : 1
}

# airline dataset specific
dataset_columns = [ 
    'Flight_Number_Reporting_Airline', 'Year', 'Quarter', 'Month', 'DayOfWeek', 
    'DOT_ID_Reporting_Airline', 'OriginCityMarketID', 'DestCityMarketID',
    'DepTime', 'DepDelay', 'DepDel15', 'ArrDel15',
    'AirTime', 'Distance' ]
target_variable = 'ArrDel15'

def train( model_params, config_params ):
    global target_variable
    rcml = rapids_cloud_ml.RapidsCloudML( model_type = config_params['model_type'],
                                          compute_type = config_params['compute'] )
        
    # ingest data
    dataset = rcml.load_data ( filename = config_params['dataset_filename'],
                                                columns = dataset_columns )

    # cross-validation [ optional ]
    cv_fold_accuracy = []
    for i_train_fold in range( config_params['CV_folds'] ):

        rcml.log( f' CV fold : { i_train_fold } \n' )

        # shuffle, split [ and persist ] data 
        X_train, X_test, y_train, y_test = rcml.split_data( dataset, target_variable, 
                                                            random_state = i_train_fold )
        
        # train model
        trained_model = rcml.train_model ( X_train, y_train, model_params )

        # evaluate perf
        test_accuracy = rcml.evaluate_test_perf ( trained_model, X_test, y_test )
        
        cv_fold_accuracy += [ test_accuracy ]
            
    # emit mean across folds
    rcml.emit_score( np.mean( cv_fold_accuracy ) )
    
    return 0
   
if __name__ == "__main__":
    config_params = rapids_cloud_ml.parse_job_name ( DEFAULT_CONFIG )
    model_params = rapids_cloud_ml.parse_model_parameters ( sys.argv[1:], config_params )    
    train ( model_params, config_params )
    sys.exit(0)