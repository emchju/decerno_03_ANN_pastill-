# -*- coding: utf-8 -*-
"""
Created on Fri Feb 04 08:37:10 2023
@author: Emelie Chandni
"""

# Import own modules
import sys
sys.path.append('../')
from model import Model
from parameters import glodbal_settings
from keras.models import load_model


def loading_process():
    loaded_model = loaded_model = load_network()
    return loaded_model
 
   
def run_AI(customer_names, customer_data): 
    print('Network loading process...')
    loaded_model = loading_process()
    print('Network loading process completed...') 
    
    for i in range(0, len(customer_data)):         
        print('------------- STARTING PREDICTION PHASE -------------')     
        current_customer_name = customer_names[i]
        current_customer_data = customer_data[i]
        model = Model(current_customer_name, current_customer_data)
        prediction_phase(model, loaded_model) 
    print('------------- ENDING PREDICTION PHASE -------------')
    

def load_network():
    if glodbal_settings['network'] == 'ann':
        network_name = glodbal_settings['model_path_ann']
        loaded_model = load_model(network_name)
        print('The pre-trained ANN network is loaded...')
        print(network_name)
    return loaded_model

            
def prediction_phase(model, loaded_network): 
    model.scaling_data()
    model.make_prediction(loaded_network)

    



    
