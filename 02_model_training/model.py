# -*- coding: utf-8 -*-
# Import libraries
import sys
sys.path.append('../')
import numpy as np
from sklearn.model_selection import train_test_split
from neural_network_clf import CLF
from neural_network_tkn import TKN
from parameters import glodbal_settings
import joblib
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

class Model:
    def __init__(self, dataset):
        
        # Training parameters
        self.epochs = glodbal_settings['epochs']
        self.batch_size = glodbal_settings['batch_size']
        
        # Project parameters
        self.dataset = dataset
        self.X = []
        self.y = []
        self.simulation_test = []
        self.data = []
        self.data_agent = []
        self.df_split_train_test = []
        self.scaler = []
        self.scaled_data = []
        self.X_train = np.array
        self.y_train = np.array
        self.X_test = np.array
        self.y_test = np.array

        
    def read_data(self):
        
        # COLUMNS in dataset:
        # --> France, Spain, Gernamy, CreditScor, Gender, Age, Tenure, 
        # --> Balance, NumOfProducts, HasCrCard, IsActiveMember, 
        # --> EstimatedSalary, Exited
        
        # X-values: col 0 --> clo -1 (France --> EstimatedSalary)
        # y-values: col -1 (Exited)
        
        self.X = self.dataset.iloc[:, 0:-1].values
        self.y = self.dataset.iloc[:, -1].values
        
        
    def data_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y, 
            test_size = glodbal_settings['test_size'], 
            random_state = 0
        )


    def text_clean(self, text_message):     
        remove_punc = [ text for text in text_message if text not in string.punctuation]    
        remove_punc = ''.join(remove_punc)    
        return [word for word in remove_punc.split() if word.lower() not in stopwords.words('english')]
       
     
    def train_model(self, network):
        if network == 'clf':
            print('Preparing network (CLASSIFIER...)')
            neural_net = CLF(self.text_clean)
            neural_net.fit(
                self.X_train, 
                self.y_train,
                )
            network_name = '../saved_model/CLF_model'
            joblib.dump(neural_net.best_estimator_, network_name)           
            print('Model saved --> ' + network_name)
        if network == 'tkn':
            print('Preparing network (TOKENIZER...)')
            neural_net = TKN(self.text_clean)
            neural_net.fit_on_texts(self.X_train)
            network_name = '../saved_model/TKN_model'
            neural_net.save(network_name)
            print('Model saved --> ' + network_name)
        return neural_net
            