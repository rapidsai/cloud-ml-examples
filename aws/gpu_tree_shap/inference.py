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
    multiple_predictions_flag = False
    
    if "csv" not in content_type:
        # request_body is a bytes object, which we decode to a string
        request_body = request_body.decode()
                    
    # request_body is of the form 'dataset, predict_function'
    # e.g. 'sklearn.datasets.fetch_california_housing(), pred_contribs'
    # comma separated: '[[var1, var2], [var3, var4], ..., varx]], pred_contribs'
    prediction_methods = ["predict", "pred_contribs", "pred_interactions"]
    if request_body.split(', ')[-1] in prediction_methods:
        if "[[" in request_body: 
            multiple_predictions_flag = True
            dataset = json.loads(", ".join(request_body.split(', ')[:-1]))
        else:
            # "var1, var2, var3, var4, ..., varx, pred_contribs"
            dataset = ", ".join(request_body.split(', ')[:-1])
            
        predict = request_body.split(', ')[-1]
    else:
        dataset = request_body
        predict = "predict"
    
    if "sklearn.datasets" in dataset:
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
    
    elif content_type == "text/libsvm":
        input_data = xgb_encoders.libsvm_to_dmatrix(dataset)
    elif content_type == "text/csv": 
        if multiple_predictions_flag:
            from pandas import DataFrame
            dataset = DataFrame(dataset)
            # this is for the NYC Taxi columns - may have to adjust for other CSV inputs
            dataset.columns = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16']
            input_data = xgb.DMatrix(dataset)
            
        else:
            input_data = xgb_encoders.csv_to_dmatrix(dataset)
        
    else: 
        raise ValueError("Content type {} is not supported.".format(content_type))
        
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
