# -*- coding: utf-8 -*-
"""
Created on Fri Feb 04 08:37:10 2022
@author: Emelie Chandni
"""
# Import libraries
import sys
sys.path.append('../')

# Import own modules
from data_preparation import DataPreparation

def read_data_from_file():    
    prepModule = DataPreparation()   
    prepModule.dataset_prep()                 
