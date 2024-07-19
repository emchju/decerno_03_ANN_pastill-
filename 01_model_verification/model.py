# -*- coding: utf-8 -*-
# Import libraries
import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from neural_network_clf import CLF
from neural_network_tkn import TKN
from parameters import glodbal_settings
import matplotlib.pyplot as plt
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
        self.error_list = [29, 30, 32, 42, 43, 44, 45, 51, 56, 63, 67, 70, 73, 80, 87, 93, 96, 102, 104]

        
    def read_data(self):
        if glodbal_settings['network'] == 'tkn':
            # Create a copy of the dataset
            dataset = pd.read_csv('../saved_data/stress.csv')
            self.dataset = dataset.copy()
        else:
            self.X = self.dataset.iloc[:, 0:-1].values
            self.y = self.dataset.iloc[:, -1].values
            
            
    def data_split(self):
        if glodbal_settings['network'] == 'tkn':
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.dataset['text'], 
                self.dataset['label'], 
                test_size = glodbal_settings['test_size'], 
                random_state = glodbal_settings['random_state']
                )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, 
                self.y, 
                test_size = glodbal_settings['test_size'], 
                random_state = glodbal_settings['random_state']
            )


    def text_clean(self, text_message):     
        remove_punc = [ text for text in text_message if text not in string.punctuation]    
        remove_punc = ''.join(remove_punc)    
        return [word for word in remove_punc.split() if word.lower() not in stopwords.words('english')]
    
    
    def clean_parameter(self, data):
        clean_texts_list = []
        print(len(data))
        for i in range(0, len(data)):           
            if i not in self.error_list:
                cleaned = self.text_clean(data[i])
                clean_texts_list.append(cleaned)
        return clean_texts_list
     
        
    def train_verifiy(self, network):
        if network == 'clf':
            print('Preparing network (CLASSIFIER...)')
            neural_net = CLF(self.text_clean)
            neural_net.fit(
                self.X_train, 
                self.y_train,
                )
        if network == 'tkn':
            print('Cleaning datasets for deep learning...')
            self.X_train = self.clean_parameter(self.X_train)
            self.X_test = self.clean_parameter(self.X_test)  
            print(self.X_train.shape)     
            print(self.X_test.shape)
            print('Preparing network (TOKENIZER...)')
            word_index, train_padded, test_padded = self.def_params()
            neural_net = TKN()
            history = neural_net.fit(
                train_padded,   # training sequence
                self.y_train, # training labels
                epochs = glodbal_settings['epochs'], 
                validation_data=(test_padded, self.y_test)
                )
            self.plot_metrics(history, "accuracy")
            self.plot_metrics(history, "loss")
        return neural_net
    
    
    def def_params(self):
        
        # Define parameters for tokenizing and padding
        trunc_type = 'post'
        oov_tok = "<oov>"
        tokenizer = Tokenizer(num_words = glodbal_settings['vocab_size'], oov_token = oov_tok)
        print(self.X_train.reshape(1,-1).shape)
        print(self.X_train[0].shape)
        tokenizer.fit_on_texts(self.X_train)
        word_index = tokenizer.word_index
        
        # Training sequences and labels
        train_seqs = tokenizer.texts_to_sequences(self.X_train)
        train_padded = pad_sequences(train_seqs, maxlen=glodbal_settings['max_length'], truncating=trunc_type)
        
        # Testing sequences and labels
        test_seqs = tokenizer.texts_to_sequences(self.X_train)
        test_padded = pad_sequences(test_seqs, maxlen=glodbal_settings['max_length'])
        return word_index, train_padded, test_padded


    def plot_metrics(history, metric):
        plt.plot(history.history[metric])
        plt.plot(history.history['val_'+ metric])
        plt.legend([metric, 'val_'+ metric])
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.show()
        
        
    def predict_verify(self, network):
        self.y_pred = network.predict(self.X_test)
        print(classification_report(self.y_test, self.y_pred))
        print(np.concatenate((self.y_pred.reshape(len(self.y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
        cm = confusion_matrix(self.y_test, self.y_pred)
        print('Confusion Matrix: ')
        print(cm) 
        print()
        print('Description Confusion Matrix:')
        print('[Predicted POSITIVE and real POSITIVE][Predicted positive and real negative]')
        print('[Predicted negative and real positive][Predicted NEGATIVE and real NEGATIVE]')  
        
        
    def accuracy(self):
        print()
        print('Accuracy Score:')
        print(accuracy_score(self.y_test, self.y_pred))
