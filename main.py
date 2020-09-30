from preprocessing import Preprocessing
from glove_vectors import Vector
from LSTM_CNN_Freezed import Freezed
from LSTM_CNN_Unfreezed import Un_freezed
from Bi_directional_LSTM import Bi_directional
import os
from os.path import abspath

import tensorflow as tf
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET ##used for XML
import keras
from tensorflow.keras import backend as K
K.clear_session()
import random, time, queue
import multiprocessing
multiprocessing.set_start_method('spawn')
from multiprocessing.managers import BaseManager
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from twisted.python.compat import raw_input
from numpy import arange
from numpy import argmax
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score

if __name__ == '__main__':

    def calc_values(probs, test_thresh):
    	return (probs >= test_thresh).astype('int')


    ## Calculates Precision, Recall and f1 score for each of the three models
    def print_values(test_data,test_values,filemodel, threshold):
    
        model = tf.keras.models.load_model(str(os.getcwd())+"\\"+"models"+'\\'+str(filemodel))

        print('\n'+str(filemodel[:-3])+": \n")
        classes = (model.predict(test_data) > threshold).astype("int32") ##Predicts the classes of the test_data

        ## Prints out Prescision, Recall , and F1-score
        precision = precision_score(test_values, classes)
        print('Precision : ' + str(precision))
        recall = recall_score(test_values, classes)
        print('Recall : '+ str(recall))
        f1 = f1_score(test_values, classes)
        print('F1 score : '+ str(f1))

        print("\n##################################")




    test_location = input("Please enter a target location for the testset. Insert the absolute path \n")
    ## A valid dataset is under (*Project folder*\en).  Remember the absolute path
    print("Processing data...")
    
    ## This part prepares the data from the test_location
    prep = Preprocessing(test_location)
    seq_dict, test_data, test_values,total_author_list = prep.main()

    train_choice = ''
    while(True):
        if (train_choice == "Y" or train_choice == "N"):
            break
        train_choice = raw_input("Do you wish to retrain the models before testing? (Y/N)  \n")

    if train_choice == 'Y':
        train_target = input("Please insert a training set location. Insert absolute path \n")
        ## A valid dataset is under (*Project folder*\Testset\). Remember the absolute path

        print("Beginning training of Bi-directional lstm")
        ## Trains the Bi_directional model
        Bi_dir = Bi_directional(train_target)
        Bi_dir.main(False)
        print("Finished training of Bi-directional lstm")
        
        print("Beginning training of LSTM/CNN with frozen weights")
        ## Trains the Freezed weigth LSTM/CNN model
        freezed = Freezed(train_target)
        freezed.main(False)
        print("Finished training of LSTM/CNN with frozen weights")
        
        print("Beginning training of LSTM/CNN with unfrozen weights")
        ## Trains the unfreezed weigth LSTM/CNN model
        un_freezed = Un_freezed(train_target)
        un_freezed.main(False)
        print("Finished training of LSTM/CNN with unfrozen weights")

    modelList = []
    for filemodel in os.listdir(str(os.getcwd())+"\\"+"models"):
        modelList.append(filemodel)

    Bimodel = str(modelList[0])
    Frozen_model = str(modelList[1])
    Unfrozen_model = str(modelList[2])


    print_values(test_data,test_values,Bimodel,0.575)
    print_values(test_data,test_values,Frozen_model,0.55)
    print_values(test_data,test_values,Unfrozen_model,0.60)



    ##This part creates and outputs the model which is used in the PAN-task. The output XML files will be in the "model_experimentation"-folder
    model = tf.keras.models.load_model(str(os.getcwd())+"\\models\\LSTM_CNN_UnFreezed.h5")
    prediction = (model.predict(test_data) > 0.60).astype("int32")

    for i in range (1,len(total_author_list)):
        root = ET.Element('author',
         author_id = total_author_list[i-1],
          lang ='en',
           type= str(prediction[i]))
        tree = ET.ElementTree(root)
        directory = ('output/')
        tree.write(str(directory)+total_author_list[i-1]+".xml")










        