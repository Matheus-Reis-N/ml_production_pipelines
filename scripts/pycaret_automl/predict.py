from pycaret.regression import *
from pycaret.classification import *
from pycaret.clustering import *

def load_model(model_source):
    if model_source == 'be_server':
        loaded_model = load_model("models://unimed_model/production")
    elif model_source == "s3":
        loaded_model = None
    else: # gcp
        loaded_model = None 
    
    return loaded_model

def predict(x_test, model_source):
    predictions = predict_model(load_model(model_source), data=x_test)