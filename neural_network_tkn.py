# -*- coding: utf-8 -*-

# Buildning the RNN (Recurrent Neural Network)
# Import Keras libraries and packages
from parameters import glodbal_settings
import keras 

def TKN():
    model = keras.Sequential([
        keras.layers.Embedding(glodbal_settings['vocab_size'], glodbal_settings['vocab_size'], input_length=glodbal_settings['max_length']),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(72, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model with loss function, optimizer and metrics
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    return model