# -*- coding: utf-8 -*-
# Import libraries
import pandas as pd

# Import own modules
from model import Model
from parameters import glodbal_settings

   
def run_AI_training(): 
    dataset = pd.read_csv(glodbal_settings['data_path'])
    model = Model(dataset)
    print('---------------- STARTING MODEL TRAINING PROCESS ----------------')      
    training_phase(model)                                         
    print('MODEL TRAINING COMPLETED...')

            
def training_phase(model):    
    model.read_data()   
    model.data_split()
    model.train_model(glodbal_settings['network'])