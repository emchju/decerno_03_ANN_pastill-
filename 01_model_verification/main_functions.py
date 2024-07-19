# -*- coding: utf-8 -*-
# Import libraries
import pandas as pd

# Import own modules
from model import Model
from parameters import glodbal_settings


def run_AI_verification(): 
    dataset = pd.read_csv(glodbal_settings['data_path'])        
    model = Model(dataset)
    print('Preparing dataset...')
    model.read_data()      
    print('Entering verification phase...')  
    verification_phase(model)                         
    print('VERIFICATION PROCESS COMPLETED...')                       
    

def verification_phase(model):
    print('Split dataset training/test...')
    model.data_split()
    print('Training and verification process in action...')
    neural_net = model.train_verifiy(glodbal_settings['network'])
    #model.predict_verify(neural_net)
    #model.accuracy()
    


    



    
