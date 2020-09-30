from preprocessing import Preprocessing
from glove_vectors import Vector
from LSTM_CNN_Freezed import Freezed
from LSTM_CNN_Unfreezed import Un_freezed
from Bi_directional_LSTM import Bi_directional
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET ##used for XML
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
K.clear_session()
from tensorflow.keras.layers import Conv1D,MaxPooling1D,Activation, Dropout, Dense, BatchNormalization, LSTM, Embedding, Reshape
from tensorflow.keras.metrics import Recall as Recall
from tensorflow.keras.metrics import Precision as Presicion
from tensorflow.keras.optimizers import Nadam as Nadam
import re
import os
from keras.regularizers import l1, l2

import random, time, queue
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score



def train_bi_directional(weight_matrix,length,train_data,train_values,test_data,test_values,dp1,dp2,dp3,n_value,dense1,dense2,lstm1,rdrop1,bsize,epoch):
    
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(length, 100, weights = [weight_matrix], trainable = False),
            tf.keras.layers.Dropout(dp1),
            tf.keras.layers.Bidirectional(LSTM(lstm1, recurrent_dropout=(rdrop1))),
            tf.keras.layers.Dense(dense1, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(loss='mean_squared_error',optimizer='nadam',metrics=['accuracy'])
        model.summary() 

        history = model.fit(train_data, train_values, batch_size=bsize, epochs=epoch, shuffle = True,validation_data=(test_data, test_values)) 
        
        writeToHistory(history,'Bi_directional',epoch,bsize,dense1,None,lstm1,dp1,None,None,rdrop1,None)


def train_freezed(weight_matrix,length,train_data,train_values,test_data,test_values,dp1,dp2,dp3,n_value,dense1,dense2,lstm1,rdrop1,bsize,epoch):
        
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(length, 100, weights = [weight_matrix], trainable = False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dp1),
            tf.keras.layers.Conv1D(4*n_value, round(4*n_value*(2/3)), activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=round(4*n_value*(2/3))),
            tf.keras.layers.Dropout(dp2),
            tf.keras.layers.Bidirectional(LSTM(lstm1,recurrent_dropout=(rdrop1))),
            tf.keras.layers.Dropout(dp3),
            tf.keras.layers.Dense(dense1, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy',optimizer='Nadam',metrics=['accuracy'])
        model.summary()

        history = model.fit(train_data, train_values, batch_size=bsize, epochs=epoch, shuffle = True,validation_data=(test_data, test_values)) 

        writeToHistory(history,'freezed',epoch,bsize,dense1,None,lstm1,dp1,dp2,dp3,rdrop1,n_value)


def train_unfreezed(weight_matrix,length,train_data,train_values,test_data,test_values,dp1,dp2,dp3,n_value,dense1,dense2,lstm1,rdrop1,bsize,epoch):

        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(length, 100, weights = [weight_matrix], trainable = True),
            tf.keras.layers.Dropout(dp1),
            tf.keras.layers.Conv1D(4*n_value, round(4*n_value*(2/3)), activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=round(4*n_value*(2/3))),
            tf.keras.layers.Dropout(dp2),
            tf.keras.layers.Dense(dense1, activation='relu'),
            tf.keras.layers.Bidirectional(LSTM(lstm1,recurrent_dropout=(rdrop1))),
            tf.keras.layers.Dense(dense2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy',optimizer='Nadam',metrics=['accuracy'])
        model.summary()

        history = model.fit(train_data, train_values, batch_size=bsize, epochs=epoch, shuffle = True,validation_data=(test_data, test_values)) 

        writeToHistory(history,'unfreezed',epoch,bsize,dense1,dense2,lstm1,dp1,dp2,None,rdrop1,n_value)


def writeToHistory(history,filename,epochs,bsize,dense1,dense2,lstm1,dp1,dp2,dp3,rdrop1,n_value):
    
    directory = "model_experimentation/"
    filelocation = directory+str(filename)
    f = open(str(filelocation),'a+')
    f.write('#########################################')
    f.write('\n')
    f.write('Accuracy: ')
    f.write(str(history.history['accuracy']))
    f.write('\n')
    f.write('Loss: ')
    f.write(str(history.history['loss']))
    f.write('\n')
    f.write('Val_accuracy: ')
    f.write(str(history.history['val_accuracy']))
    f.write('\n')
    f.write('val_loss: ')
    f.write(str(history.history['val_loss'])) 
    f.write('\n')
    f.write('\n')

    f.write("Epochs: " + str(epochs))
    f.write('\n')
    f.write("bsize: " + str(bsize))
    f.write('\n')
    f.write("lstm1: " + str(lstm1))
    f.write('\n')
    if n_value is not None:
        f.write("n_value: " + str(n_value))
        f.write('\n')
    if rdrop1 is not None:
        f.write("rdrop: " + str(rdrop1))
        f.write('\n')
    if dp1 is not None:
        f.write("dp1: " + str(dp1))
        f.write('\n')
    if dp2 is not None:
        f.write("dp2: " + str(dp2))
        f.write('\n')
    if dp3 is not None:
        f.write("dp3: " + str(dp3))
        f.write('\n')
    if dense1 is not None:
        f.write("dense1: " + str(dense1))
        f.write('\n')
    if dense2 is not None:
        f.write("dense2: " + str(dense2))
        f.write('\n')
    f.write('#########################################')

    f.close()

def get_random_values():
    n_value = random.randint(1,3)
    epoch = random.randint(5,15)
    dp1 = (random.randint(1,3))/10
    dp2 = (random.randint(1,3))/10
    dp3 = (random.randint(1,3))/10
    bsize = (random.randint(5,15))
    rdrop1 = (random.randint(1,3))/10
    dense1 = random.randint(1,8)*16
    dense2 = dense1/4
    lstm1 = random.randint(10,20)
    return epoch, bsize, dense1, dense2, lstm1, dp1, dp2, dp3 ,rdrop1, n_value

if __name__ == '__main__':
    
    prep = Preprocessing('en')
    seq_dict, text_list, value_list,total_author_list = prep.main()
    GloVe = Vector(seq_dict)
    weight_matrix = GloVe.return_weights(seq_dict)
    train_data,test_data,train_values,test_values = train_test_split(text_list,value_list,test_size =0.1 ,shuffle = True)
    length = len(seq_dict)+1

    iteration = 0
    while iteration < 30:
        epoch, bsize, dense1, dense2, lstm1, dp1, dp2, dp3 ,rdrop1, n_value = get_random_values()
        train_unfreezed(weight_matrix,length,train_data,train_values,test_data,test_values,dp1,dp2,dp3,n_value,dense1,dense2,lstm1,rdrop1,bsize,epoch)
    
        epoch, bsize, dense1, dense2, lstm1, dp1, dp2, dp3 ,rdrop1, n_value = get_random_values()
        train_freezed(weight_matrix,length,train_data,train_values,test_data,test_values,dp1,dp2,dp3,n_value,dense1,dense2,lstm1,rdrop1,bsize,epoch)
        
        epoch, bsize, dense1, dense2, lstm1, dp1, dp2, dp3 ,rdrop1, n_value = get_random_values()
        train_bi_directional(weight_matrix,length,train_data,train_values,test_data,test_values,dp1,dp2,dp3,n_value,dense1,dense2,lstm1,rdrop1,bsize,epoch)
        iteration = iteration + 1