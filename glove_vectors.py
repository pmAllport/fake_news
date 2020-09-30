import numpy as np
class Vector():

    #########################################################
    ## This class takes in a set() of words and creates a  ##
    ## weight_matrix out of those words with GloVe-vectors.##
    #########################################################

    def return_weights(self,seq_dict):
        size = len(seq_dict)+1 # Used to create the weightmatrix
        embedding_dict = dict()
        glove_vectors = open('glove.6B.100d.txt','r', encoding="utf8") 

        for line in glove_vectors.readlines():

            ##This part splits the words from the values,
            ##as the textfile is laid out as [*Word*,*Value*,*Value*,*Value*,...,*Value*]
            vector_line = line.split(" ")
            targetword = vector_line[0]
            templist = vector_line[1:]


            ##Converts values to ndarray and puts it in embedding_dict
            temp_np = np.asarray(templist)
            embedding_dict[targetword] = temp_np

        ##Creates a large emtpy matrix used for inserting values later on
        weight_matrix = np.zeros((size, 100))


        ##This checks if the key if GloVe has a vector that it is able to use instead of the 0
        ##and inserts it if GloVe has one
        for word in seq_dict.keys():
            if word in embedding_dict.keys():
                index = int(seq_dict[word])
                weight_matrix[index] = embedding_dict.get(word)


        return weight_matrix



        #####################################
        ##                                 ##
        ## The output will look like this: ##
        ## [[0,0,0,0,0,0,0,0, ... ,0]      ##
        ##  [0.33,0.11,0.22, ... , 0.6]    ##
        ##  .....................          ##
        ##  [0.33,0.11,0.22, ... , 0.6]]   ##
        ##                                 ##
        #####################################

    def __init__(self,seq_dict):
        self.seq_dict = seq_dict
        self.return_weights(self.seq_dict)