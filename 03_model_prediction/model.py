# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 07:03:27 2023

@author: Emelie Chandni
"""

# Import libraries
import sys
sys.path.append('../')
import pickle
from parameters import glodbal_settings

class Model:
    def __init__(self, customer_name, customer_data):
        
        # Training parameters
        self.epochs = glodbal_settings['epochs']
        self.batch_size = glodbal_settings['batch_size']
        self.scaler = pickle.load(open(glodbal_settings['scaler_path'], 'rb'))
        
        # Project parameters
        self.customer_name = customer_name
        self.customer_data = customer_data
        self.customer_scaled = []
        self.pred = []
  
            
    def scaling_data(self):       
        self.customer_scaled = self.scaler.transform(self.customer_data)
        print('Data for customer is scaled...')
           
             
    def make_prediction(self, loaded_model): 
        print('Making prediction for customer ' + self.customer_name)
        result = loaded_model.predict(self.customer_scaled)
        bool_test = result > 0.5    
        print('Results for customer ' + self.customer_name + ':')
        print('Result: ', result)
        print('Result: ', result > 0.5)        
        if bool_test == True:
            print('--> This customer will most likely LEAVE the bank.')
        if bool_test == False:
            print('--> This customer will most likely STAY as customer on the bank.')
