import os
import xml.etree.ElementTree as ET ##used for XML
import nltk
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = list(set(stopwords.words("english")))



class Preprocessing():
##############################################################################################################
##THIS PART EXTRACTS FROM DIRECTORY


    ## This function reads the "Truth.txt" and creates two lists of which unique identifiers
    ## are true and which are false
    ## The lines in this look like  "nmdasdnbksadnajdnkdnadnkj:::1" in which the part before
    ## the ":::" are the unique identifier (and also the name of the corresponding file)
    ## and the latter are the truth-value

    def gettruth(self,filename):
        total_author_list = []
        truth_list = []
        lie_list = []
        file_truth = open(filename, "r").readlines()
        for line in file_truth:
            line = line.rstrip("\n")
            split_string = line.split(":::")
            if split_string[1] == "1":
                truth_list.append(str(split_string[0]))
                total_author_list.append(str(split_string[0]))
            elif split_string[1] == "0":
                lie_list.append(str(split_string[0]))
                total_author_list.append(str(split_string[0]))
        return truth_list, lie_list,total_author_list


    ## A lot of these Tweets contain non-ascii characters like greek symbols and emojis
    ## in which there is no current implementation to handle it
    def ASCIIFy(self,inputString):
        return inputString.encode('ascii', 'ignore').decode('ascii')

    ## Lowers and lemmatizes words if they are not in the stopwords
    def preprocess(self,text):
        words = word_tokenize(text)
        lemmatized = [lemmatizer.lemmatize(w.lower()) for w in words if w not in stop_words]
        return lemmatized

    ## Gets the sentences from the .xml files and puts them in a dictionary in which the unique identifier
    ## is the key. This is so that one is able to link the sentences to their truth value later on.

    def getsents(self,directory):
        text_dict = dict()
        word_list = []
        for file in os.listdir(directory):
            if file.endswith(".xml"):
                templist = []
                tree = ET.parse(str(directory)+"/"+str(file))
                root = tree.getroot()
                for element in root[0]:
                    processedtext = self.preprocess(self.ASCIIFy(element.text))
                    word_list.extend(processedtext)
                    templist.append(processedtext)

                text_dict[file[:len(file)-4]] = templist
        return text_dict, word_list

##THIS PART EXTRACTS FROM DIRECTORY
##############################################################################################################
##THIS PART CONVERTS FROM WORDS TO INTEGERS AND PADS/SPLICES THE SENTS
    

    ## This creates a set of all the words in the sentences. In addition we append a "0" so in case we meet
    ## an unknown word further on when analysing a test-set, we are able to give it an integer value of this
    ## "0".
    def word_seq(self,word_list): 
        templist = []
        tempdict = dict()
        templist.append("0")
        templist.extend(list(set(word_list)))
        for word in templist:
            tempdict[str(word)] = templist.index(word)
        return tempdict 


    ## This checks if the length of the sentence is shorter or longer than the length-parameter and
    ## Pads/Splices the sentence to the correct length
    ## Padded sentences get [0]'s at the end of them
    def pad_limit(self,sent,length): 
        if len(sent) > length:
            return sent[:length]
        elif len(sent) < length:
            difference = length - len(sent)
            pad_list = difference * [0]
            return sent+pad_list
        else:
            return sent


    ## Takes in a sentence and converts each word to its corresponding unique value from the "seq_dict" set
    ## A words unique value is equal to their index in this "set" (Actually a list as you need to get indexes)
    ## If a word has not been seen before, it is set to the index of 0
    def numerize(self,sent,seq_dict):
        templist = []
        for word in sent:
            try:
                templist.append(seq_dict[word])
            except ValueError:
                templist.append(seq_dict["0"])
        templist = self.pad_limit(templist,15)
        return templist

    ##This is the "main" function of this subsection which converts from words to integers
    ## It takes all the sentences, from every person and does three things:
    ## * Converts the words to their respective integers
    ## * Checks if the sentence needs to be padded/spliced and returns one with a valid length
    ## * For each sentence, it appends a 1 or a 0 to "valuelist. The valuelist indicates
    ## if a sentence belongs to a person who has created fake news or not. 0 means liar, 1 means truther(?)
    def fill_text_values(self,texts,truth_list,lie_list,seq_dict):
        text_list = []
        value_list = []
        for key in texts.keys():
            for sent in texts[key]:
                text_list.append(self.numerize(sent,seq_dict))
                if key in truth_list:
                    value_list.append(1)
                elif key in lie_list:
                    value_list.append(0)
        return text_list, value_list

##THIS PART CONVERTS FROM WORDS TO INTEGERS AND PADS/SPLICES THE SENTS
##############################################################################################################
    
    ##################################################################################################################
    ## This class takes in a directory location and outputs 4 things:                                               ##                                                                    ##
    ## * A dicitonary of all the words in the corpus. Their values correspond to their unique word ID (seq_dict)    ##
    ## * A textlist of all the sentences in the tweets. They are all tokenized and lemmatized                       ##
    ##   and converted to their respective integer values                                             (text_list)   ##
    ## * A valuelist indicating if the sentence came from a liar or not a liar                        (value_list)  ##
    ## * A list of all the authors in the order they were added                                (total_author_list)  ##
    ##                                                                                                              ##
    ## The output looks like this:                                                                                  ##
    ## [[32131,0,432,543223,432, ... , 2131]                                                                        ##
    ## [1,563,3,41,342,431,4233, ... , 4131]                                                                        ##  
    ##   .................................                                                                          ##
    ## [43,1231,5334,54354353,0, ... ,5345]]                                                                        ##
    ##                                                                                                              ##
    ##################################################################################################################
    


    def main(self):
        truth_list, lie_list,total_author_list = self.gettruth(self.location+"/truth.txt")
        text_dict, word_list = self.getsents(self.location)
        seq_dict = self.word_seq(word_list)
        text_list,value_list = self.fill_text_values(text_dict,truth_list,lie_list,seq_dict)

        return seq_dict, text_list, value_list,total_author_list
    
    def __init__(self,location):
        self.location = location