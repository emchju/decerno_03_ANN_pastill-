# -*- coding: utf-8 -*-
import sys
sys.path.append('../')
import pandas as pd
from parameters import glodbal_settings

import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


class DataPreparation:
    def __init__(self):
        self.X = []
        self.y = []
        self.column_names = [           
            'Text'
            ]
            
    def dataset_prep(self):
        dataset = pd.read_csv('../saved_data/stress.csv')
        dataset['label_in_value'] = dataset['label'].map({0:'No Stress',1:"Stress"})
        
        texts = dataset['text']
        clean_texts_list = []
        for i in range(0, len(texts)):
            cleaned = self.text_clean(texts[i])
            clean_texts_list.append(cleaned)
        
        dataset['text'] = clean_texts_list
        
        self.X = dataset['text'].values
        self.y = dataset['label_in_value'].values
                
        dataset_prep = pd.DataFrame(self.X)
        dataset_prep.columns = self.column_names
        dataset_prep  = dataset_prep.assign(Stress = self.y)
        dataset_prep.to_csv(glodbal_settings['data_path'], index = False)


    def text_clean(self, text_message):     
        remove_punc = [ text for text in text_message if text not in string.punctuation]    
        remove_punc = ''.join(remove_punc)    
        return [word for word in remove_punc.split() if word.lower() not in stopwords.words('english')]


        