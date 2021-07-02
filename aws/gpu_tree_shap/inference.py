import json
import os
import pickle as pkl

import numpy as np
import sagemaker_xgboost_container.encoder as xgb_encoders
import xgboost as xgb


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    model_file = "xgboost-model"
    booster = pkl.load(open(os.path.join(model_dir, model_file), "rb"))
    return booster


def transform_fn(model, request_body, content_type, accept_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type, 
    we first need to create a DMatrix (an object that can be passed to predict)
    """
    
    # request_body is a bytes object, which we decode to a string
    request_body = request_body.decode()
    # request_body is of the form 'dataset, predict_function'
    # e.g. 'sklearn.datasets.fetch_california_housing(), pred_contribs'
    dataset = request_body.split(', ')[0]
    try:
        predict = request_body.split(', ')[1]
    except IndexError: 
        predict = "predict"
    
    if "sklearn.datasets" in request_body:
        import sklearn.datasets
        
        try: 
            data = eval(dataset)
        except Exception: 
            raise ValueError("Function {} is not supported. Try something like 'sklearn.datasets.fetch_california_housing()'"
                             .format(dataset))
            
        X = data.data
        y = data.target
        dmat = xgb.DMatrix(X, y)
        input_data = dmat
    elif request_content_type == "text/libsvm":
        input_data = xgb_encoders.libsvm_to_dmatrix(dataset)
    else: 
        raise ValueError("Content type {} is not supported.".format(request_content_type))
        
    """
    Now that we have the DMatrix and a prediction method, 
    we invoke the predict method and return the output. 
    """
    if "predict" in predict: 
        predictions = model.predict(input_data)
        return str(predictions.tolist())
    
    elif "pred_contribs" in predict: 
        shap_values = model.predict(input_data, pred_contribs=True)
        return str(shap_values.tolist())
        
    elif "pred_interactions" in predict: 
        shap_interactions = model.predict(input_data, pred_interactions=True)
        return str(shap_interactions.tolist())
    
    else: 
        raise ValueError("Prediction parameter {} is not supported.".format(predict))
