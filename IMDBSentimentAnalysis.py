#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Dec 14 19:52:40 2018
@author: samirkarmacharya
"""

import pandas as pd
from pandas import DataFrame,read_csv
import os
import csv
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.linear_model import SGDClassifier 




training_dataset = r"/Users/samirkarmacharya/Documents/UIUC/CS410/Project/IMDB Dataset/aclImdb/train/"
testing_dataset = r"/Users/samirkarmacharya/Documents/UIUC/CS410/Project/IMDB Dataset/aclImdb/test/" 


'''
IMDB_data_preprocess function explores the neg and pos folders from aclImdb/train and aclImdb/test folders and creates the training_dataset and testing_dataset csv combining both positive and negative 
reviews respectively. Training data is created after removing the stopwords, whereas testing data is created after removing stopwords and polarity.
'''
def imdb_data_preprocess(inpath, outpath="./", name="training_dataset.csv", mix=False, train_data = True):


    stopwords = open("stopwords.txt", 'r' , encoding="ISO-8859-1").read()
    stopwords = stopwords.split("\n")

    indices = []
    text = []
    rating = []

    i =  0 


    for filename in os.listdir(inpath+"pos"):
        data = open(inpath+"pos/"+filename, 'r' , encoding="ISO-8859-1").read()
        data = remove_stopwords(data, stopwords)
        indices.append(i)
        text.append(data)
        if train_data:
            rating.append("1")
        i = i + 1

    for filename in os.listdir(inpath+"neg"):
        data = open(inpath+"neg/"+filename, 'r' , encoding="ISO-8859-1").read()
        data = remove_stopwords(data, stopwords)
        indices.append(i)
        text.append(data)
        if train_data:
            rating.append("0")
        i = i + 1

    if train_data:
        Dataset = list(zip(indices,text,rating))
    else:
        Dataset = list(zip(indices,text))
    
    #if mix:
    np.random.shuffle(Dataset)
    if train_data:
        df = pd.DataFrame(data = Dataset, columns=['row_Number', 'text', 'polarity'])
        df.to_csv(outpath+name, index=False, header=True)
    else:
        df = pd.DataFrame(data = Dataset, columns=['row_Number', 'text'])
        df.to_csv(outpath+name, index=False, header=True)

    pass


#remove stopwords
def remove_stopwords(sentence, stopwords):
    sentencewords = sentence.split()
    resultwords  = [word for word in sentencewords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result


#apply unigram
def unigram_process(data):
    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(data)
    return vectorizer   


#apply bigram
def bigram_process(data):
    vectorizer = CountVectorizer(ngram_range=(1,2))
    vectorizer = vectorizer.fit(data)
    return vectorizer


#apply tfidf
def tfidf_process(data):
    transformer = TfidfTransformer()
    transformer = transformer.fit(data)
    return transformer


'''
RETRIEVE_DATA takes a CSV file as the input and returns the corresponding arrays of labels and data as output. 
Name - Name of the csv file 
Train - If train is True, both the data and labels are returned. Else only the data is returned 
'''
def retrieve_data(name="training_dataset.csv", train=True):
    data = pd.read_csv(name,header=0, encoding = 'ISO-8859-1')
    X = data['text']

    if train:
        Y = data['polarity']
        return X, Y
    

    return X        



'''
STOCHASTIC_DESCENT applies Stochastic on the training data and returns the predicted labels 
Xtrain - Training Data
Ytrain - Training Labels
Xtest - Test Data 
'''
def stochastic_descent(Xtrain, Ytrain, Xtest):
    clf = SGDClassifier(loss="hinge", penalty="l1", n_iter=20)
    print ("SGD Fitting")
    clf.fit(Xtrain, Ytrain)
    print ("SGD Predicting")
    Ytest = clf.predict(Xtest)
    return Ytest


'''
Predict the accuracy of the algorithms on testing and training datasets.
Ytrain - One set of labels 
Ytest - Other set of labels 
'''
def accuracy(Ytrain, Ytest):
    assert (len(Ytrain)==len(Ytest))
    num =  sum([1 for i, word in enumerate(Ytrain) if Ytest[i]==word])
    n = len(Ytrain)  
    return (num*100)/n




'''Write_txt function reads the testing data and outputs a csv file with
columns, ID, Reiview, Polarity'''
def write_txt(data, name, inpath = "./testing_dataset.csv"):
    import pandas as pd
    #data = ''.join(str(word) for word in data)
    DataFrame = pd.read_csv(inpath,header=0, encoding = 'ISO-8859-1')
    DataFrame["polarity"]= data
    DataFrame.to_csv(name, sep= ',', index = False, header =['Row_Num','Review','Polarity'])
    pass 



if __name__ == "__main__":
    import time
    start = time.time()
    print ("Preprocessing the training data...")
    imdb_data_preprocess(inpath=training_dataset, mix=True)
    print ("Preprocessing of the training datais complete. Retreiving the training data...")
    [Xtrain_text, Ytrain] = retrieve_data()
    print ("Retrieved the training data. Creating the testing data...")
    imdb_data_preprocess(inpath=testing_dataset,outpath="./", name="testing_dataset.csv", mix=False, train_data=False)
    print("Created test data. Retreiving testing data....")
    Xtest_text = retrieve_data(name="testing_dataset.csv", train=False)
    print ("Retrieved the test data. Initializing the model...\n\n")
    


    print ("-----------------------ANALYSIS ON TRAINING DATA---------------------------")
    uni_vectorizer = unigram_process(Xtrain_text)
    print ("Fitting the unigram model")
    Xtrain_uni = uni_vectorizer.transform(Xtrain_text)
    print ("Fitting of the unigram model complete.")
    print ("Applying the stochastic descent")
    Y_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtrain_uni)
    print ("Stochastic descent applied")
    print ("Accuracy for the Unigram Model is ", accuracy(Ytrain, Y_uni))
    print ("\n")

    bi_vectorizer = bigram_process(Xtrain_text)
    print ("Fitting the bigram model")
    Xtrain_bi = bi_vectorizer.transform(Xtrain_text)
    print ("Fitting of the bigram model complete")
    print ("Applying the stochastic descent")
    Y_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtrain_bi)
    print ("Stochastic descent applied")
    print ("Accuracy for the Bigram Model is ", accuracy(Ytrain, Y_bi))
    print ("\n")

    uni_tfidf_transformer = tfidf_process(Xtrain_uni)
    print ("Fitting the tfidf for unigram model")
    Xtrain_tf_uni = uni_tfidf_transformer.transform(Xtrain_uni)
    print ("Fitting of the tfidf for unigram model complete")
    print ("Applying the stochastic descent")
    Y_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtrain_tf_uni)
    print ("Stochastic descent applied")
    print ("Accuracy for the Unigram TFIDF Model is ", accuracy(Ytrain, Y_tf_uni))
    print ("\n")


    bi_tfidf_transformer = tfidf_process(Xtrain_bi)
    print ("Fitting the tfidf for bigram model")
    Xtrain_tf_bi = bi_tfidf_transformer.transform(Xtrain_bi)
    print ("Fitting of the tfidf for bigram model complete")
    print ("Applying the stochastic descent")
    Y_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtrain_tf_bi)
    print ("Stochastic descent applied")
    print ("Accuracy for the BIgram TFIDF Model is ", accuracy(Ytrain, Y_tf_bi))
    print ("\n")


    print ("-----------------------ANALYSIS ON THE TESTING DATA ---------------------------")
    print ("Unigram Model on the Test Data--")
    Xtest_uni = uni_vectorizer.transform(Xtest_text)
    print ("Applying the stochastic descent")
    Ytest_uni = stochastic_descent(Xtrain_uni, Ytrain, Xtest_uni)
    write_txt(Ytest_uni, name="unigram.output.csv")
    print ("Stochastic descent applied")
    print ("\n")


    print ("Bigram Model on the Test Data--")
    Xtest_bi = bi_vectorizer.transform(Xtest_text)
    print ("Applying the stochastic descent")
    Ytest_bi = stochastic_descent(Xtrain_bi, Ytrain, Xtest_bi)
    write_txt(Ytest_bi, name="bigram.output.csv")
    print ("Stochastic descent applied")
    print ("\n")

    print ("Unigram TF Model on the Test Data--")
    Xtest_tf_uni = uni_tfidf_transformer.transform(Xtest_uni)
    print ("Applying the stochastic descent")
    Ytest_tf_uni = stochastic_descent(Xtrain_tf_uni, Ytrain, Xtest_tf_uni)
    write_txt(Ytest_tf_uni, name="unigramtfidf.output.csv")
    print ("Stochastic descent applied")
    print ("\n")

    print ("Bigram TF Model on the Test Data--")
    Xtest_tf_bi = bi_tfidf_transformer.transform(Xtest_bi)
    print ("Applying the stochastic descent")
    Ytest_tf_bi = stochastic_descent(Xtrain_tf_bi, Ytrain, Xtest_tf_bi)
    write_txt(Ytest_tf_bi, name="bigramtfidf.output.csv")
    print ("Stochastic descent applied")
    print ("\n")

    
    print ("Total time taken is ", time.time()-start, " seconds")
    pass
