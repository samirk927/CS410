# CS410
IMDB Review Sentiment Analysis

IMDB or Internet Movie Database provides diverse information on Movies, TV series and their associated cast, crew, biographies, trivia etc. One of the most interesting and important information that it provides is reviews and ratings. The objective of this project is to train the model and predict the sentiment on the testing dataset.

Dataset for this project has been gathered from:
Source:

http://ai.stanford.edu/~amaas/data/sentiment/

The dataset contains 50,000 reviews from IMDB which are divided evenly into 25K training and 25K testing dataset. On the entire collection, no more than 30 reviews are allowed per movie. Any movie with a rating score of <=4 is considered as negative and >=7 as positive. Neutral ratings(rating 5 and 6) are not considered.

Files

There are two main directories (train/ and test/) which corresponds to training and testing dataset. They include both positive and negative reviews. 

Environment:
Python 3

Libraries: 

Pandas, Scikit,Numpy

Overview of functions:

1)	imdb_data_preprocess : Creates training dataset as part of preprocessing where the stop words are removed and stored in training dataset path:

• training_dataset = "./aclImdb/train/" # source data
Testing dataset is created as part of preprocessing where the stop words and polarity are removed and stored in testing dataset path:
• testing_dataset = "./aclImdb/test/"

2)	remove_stopwords : Removes the stopwords from the input sentence and returns the sentence.

3)	 unigram_process : Takes the data as the input and returns a vectorizer of the unigram as output

4)	bigram_process : Takes the data as the input and returns a vectorizer of the bigram as output

5)	tfidf_process : Takes the data as the input and returns a vectorizer of the tfidf as output

6)	retrieve_data : Takes a CSV file as the input and returns the corresponding arrays of labels and data as output

7)	stochastic_descent : Applies Stochastic descent on the training data and returns the predicted labels

8)	accuracy : Finds the accuracy in percentage given the training and test labels

9)	write_txt : Takes training/testing data as input and writes to a csv file.


Step 1: Reading and preprocessing the data:

Download the data from mentioned source and unzip.

Download the stopwords.txt from:

https://github.com/samirk927/CS410

Training dataset is created as part of preprocessing where the stop words are removed and stored in training dataset path:

•	training_dataset = "./aclImdb/train/" # source data

Testing dataset is created as part of preprocessing where the stop words and polarity are removed and stored in testing dataset path:

•	testing_dataset = "./aclImdb/test/" 

Step 2- Algorithms used:

For the purposed of this project unigram, bigram and tfidf algorithms are used to analyze the data.

Step 3- Classifier 

Stochastic Gradient Descent classifier is used in this project to minimize the processing expense instead of gradient descent, as gradient descent tends to be expensive on large dataset. 


Step 4-Analysis on training data:

Accuracy of unigram model is 92.604

Accuracy for the Bigram Model is 93.952

Accuracy for the Unigram TFIDF Model is 88.344

Accuracy for the Bigram TFIDF Model is 86.16




Step 5- Applying the classifier on testing data:

Four files are outputted:

1.	bigram.output.csv
2.	bigramtfidf.output.csv
3.	unigram.output.csv
4.	unigramtfidf.output.csv 


Step 6- Reviewing the results:

Sample from bigram.output.csv, where the polarity has been predicted based on the testing data. 
(Please see https://github.com/samirk927/CS410/blob/master/Samir_KarmacharyaCS%20410%20Course%20Project.pdf)

Step 7: The code file:

Run:

https://github.com/samirk927/CS410/blob/master/IMDBSentimentAnalysis.py


Performance:

In my testing with 25K training and 25K testing dataset, the code ran for ~100 seconds.



