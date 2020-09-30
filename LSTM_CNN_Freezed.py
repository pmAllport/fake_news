from preprocessing import Preprocessing
from glove_vectors import Vector
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

class Freezed():
    
    def train_model(self,weight_matrix,length,train_data,train_values,test_data,test_values):
        
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(length, 100, weights = [weight_matrix], trainable = False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(8, 4, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=4),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Bidirectional(LSTM(10,recurrent_regularizer=l2(1e-4))),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy',optimizer='Nadam',metrics=['accuracy'])
        model.summary()

        history = model.fit(train_data, train_values, batch_size=10, epochs=20, shuffle = True,validation_data=(test_data, test_values)) 

        model.save(str(os.getcwd())+"/"+"models"+'LSTM_CNN_Freezed.h5', model)


        return history


    #############################################################################
    ## THIS PART IS THE SAME IN EVERY MODEL

    def plot_model(self,history):
        plt.plot(history.history['accuracy'])## Plots accuracy
        plt.plot(history.history['loss']) ## Plots Loss
        plt.plot(history.history['val_accuracy']) ## Plots Valication accuracy
        plt.plot(history.history['val_loss']) ## Plots Valication Loss
        plt.title('Model Accuracy/Loss | Validation Accuracy/loss') ## Creates Title
        plt.ylabel('Accuracy') ## sets Y-label
        plt.xlabel('Epoch') ## Sets X label
        plt.legend(['Accuracy', 'Loss','Val_accuracy','Val_loss',], loc='upper left') ## Creates legend
        plt.show() ## Shows plot

        
    def main(self,do_plot):
        ## This part creates a preprocessing instance. What it returns and what they signify is explained in preprocessing.py
        prep = Preprocessing(self.location)
        seq_dict, text_list, value_list,total_author_list = prep.main() ## Total author list is used in evaluation

        ## This part creates a Vector instance and returns a GloVe weight matrix
        GloVe = Vector(seq_dict)
        weight_matrix = GloVe.return_weights(seq_dict)

        ##Splits the data into test and train sets
        train_data,test_data,train_values,test_values = train_test_split(text_list,value_list,test_size =0.125 ,shuffle = True)
        
        ## The training of the model
        history = self.train_model(weight_matrix,len(seq_dict)+1,train_data,train_values,test_data,test_values)
        
        ##This is if you want the plot the history
        if(do_plot == True):
            self.plot_model(history)
    
    def __init__(self,location):
        self.location = location
    
    ## THIS PART IS THE SAME IN EVERY MODEL
    #############################################################################

if __name__ == '__main__':
    Freezed = Freezed('en')
    Freezed.main(True)        
