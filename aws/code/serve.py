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

import os, sys, traceback
import joblib 
import glob
import json
import time

import xgboost
import numpy

import flask 
from flask import Flask, Response

try:
    """ check for GPU via library imports """
    import cupy
    from cuml import ForestInference
    GPU_INFERENCE_FLAG = True

except Exception as gpu_import_error:
    GPU_INFERENCE_FLAG = False
    print( f'\n!GPU import error: {gpu_import_error}\n')

# set to true to print incoming request headers and data
DEBUG_FLAG = False

def serve( xgboost_threshold = 0.5 ):
    """ Simple Flask Inference Server for SageMaker hosting of RAPIDS Models """
    app = Flask(__name__)

    if GPU_INFERENCE_FLAG:
        print( 'GPU Model Serving Workflow')
        print( f'> {cupy.cuda.runtime.getDeviceCount()} GPUs detected \n')
    else:
        print( 'CPU Model Serving Workflow')
        print( f'> {os.cpu_count()} CPUs detected \n')
        
    @app.route("/ping", methods=["GET"])
    def ping():
        """ SageMaker required method for ensuring serving instance has a heartbeat """
        return Response(response="\n", status=200)
    
    def load_trained_model ():
        """ 
        Load [ XGBoost or RandomForest ] model into memory 
        This function scans the model SageMaker model directory and parses saved model names
        """
        xgb_models = glob.glob('/opt/ml/model/*_xgb')
        rf_models = glob.glob('/opt/ml/model/*_rf')
        print( f'detected xgboost models : {xgb_models}' ) 
        print( f'detected randomforest models : {rf_models}\n\n' ) 
        model_type = None

        start_time = time.time()

        if len( xgb_models ):
            model_type = 'XGBoost'
            model_filename = xgb_models[0]
            if GPU_INFERENCE_FLAG:
                # FIL 
                reloaded_model = ForestInference.load( model_filename )
            else:
                # native XGBoost
                reloaded_model = xgboost.Booster()                
                reloaded_model.load_model ( fname = model_filename )
        elif len( rf_models):
            model_type = 'RandomForest'
            model_filename = rf_models[0]
            reloaded_model = joblib.load( model_filename )
        else:
            raise Exception ('! No trained models detected')

        print(f'> model {model_filename} loaded in { round( time.time() - start_time, 5) } seconds \n')

        return reloaded_model, model_type, model_filename

    @app.route("/invocations", methods=["POST"])
    def predict():
        """ Run CPU or GPU inference on input data, called everytime an incoming request arrives """

        # parse user input
        try:
            if DEBUG_FLAG:
                print( flask.request.headers )
                print( flask.request.content_type )
                print( flask.request.get_data() )

            string_data = json.loads( flask.request.get_data() )
            deserialized_data = numpy.array ( string_data ) # , order = 'F' 

        except Exception as input_parsing_error:
            return Response ( response = f"Unable to parse input data [ should be json/string encoded list of arrays ] \n",
                              status = 415, mimetype='text/csv')

        # load trained model and run Flask server to process incoming requests
        reloaded_model, model_type, model_filename = load_trained_model()
        
        # run inference
        try:
            start_time = time.time()
            if model_type == 'XGBoost':
                print(f'running inference using XGBoost model : {model_filename}')
                
                if GPU_INFERENCE_FLAG:
                    predictions = reloaded_model.predict( deserialized_data )
                else:
                    dmatrix_deserialized_data = xgboost.DMatrix( deserialized_data )
                    predictions = reloaded_model.predict( dmatrix_deserialized_data )

                predictions = ( predictions > xgboost_threshold ) * 1.0

            elif model_type == 'RandomForest':
                print(f'running inference using RandomForest model : {model_filename}')

                if 'gpu' in model_filename and GPU_INFERENCE_FLAG == False:
                    raise Exception( 'attempting to run CPU inference on a GPU trained RandomForest model')

                # this should fail on GPU trained models 
                predictions = reloaded_model.predict( deserialized_data.astype('float32') )

            print(f'\n predictions: {predictions} \n')
            print(f' > inference finished in { round( time.time() - start_time, 5) } seconds \n')

            # return predictions
            return Response( response = json.dumps( predictions.tolist() ),
                             status=200, mimetype='text/csv' )
        
        # error during inference 
        except Exception as inference_error:
            print( inference_error )
            return Response ( response = f"Inference failure: {inference_error}\n", 
                              status = 400, mimetype='text/csv')
        
    app.run(host="0.0.0.0", port=8080)
    
if __name__ == "__main__":

    try : 
        serve ()
        sys.exit(0) # success exit code

    except Exception as error:        
        trc = traceback.format_exc()           
        print( f' ! exception: {str(error)} \n {trc}')
        sys.exit(-1) # failure exit code

"""
simple airline model inference test [ 3 non-late flights, and a one late flight ]
curl -X POST --header "Content-Type: application/json" --data '[[ 2019.0, 4.0, 12.0, 2.0, 3647.0, 20452.0, 30977.0, 33244.0, 1943.0, -9.0, 0.0, 75.0, 491.0 ], [0.6327389486117129, 0.4306956773589715, 0.269797132011095, 0.9802453595689266, 0.37114359481679515, 0.9916185580669782, 0.07909626511279289, 0.7329633329905694, 0.24776047025280235, 0.5692037733986525, 0.22905629196095134, 0.6247424302941754, 0.2589150304037847], [0.39624412725991653, 0.9227953615174843, 0.03561991722126401, 0.7718573109543159, 0.2700874862088877, 0.9410675866419298, 0.6185692299959633, 0.486955878112717, 0.18877072081876722, 0.8266565188148121, 0.7845597219675844, 0.6534800630725327, 0.97356320515559], [ 2018.0, 3.0, 9.0, 5.0, 2279.0, 20409.0, 30721.0, 31703.0, 733.0, 123.0, 1.0, 61.0, 200.0 ]]' http://0.0.0.0:8080/invocations
"""