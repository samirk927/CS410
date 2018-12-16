# CS410
IMDB Review Sentiment Analysis

IMDB or Internet Movie Database provides diverse information on Movies, TV series and their associated cast, crew, biographies, trivia etc. One of the most interesting and important information that it provides is reviews and ratings. The objective of this project is to train the model and predict the sentiment on the testing dataset.

Dataset for this project has been gather from:
Source:
http://ai.stanford.edu/~amaas/data/sentiment/

The dataset contains 50,000 reviews from IMDB which are divided evenly into 25K training and 25K testing dataset. For the purpose of this project the testing dataset has been created by removing the polarity from the training dataset. On the entire collection, no more than 30 reviews are allowed per movie. Any movie with a rating score of <=4 is considered as negative and >=7 as positive. Neutral ratings(rating 5 and 6) are not considered.

Files

There are two main directories (train/ and test/) which corresponds to training and testing dataset. They include both positive and negative reviews. 

Step 1: Reading and preprocessing the data:

Training dataset is created as part of preprocessing where the stop words are removed and stored in training dataset path:
•	training_dataset = "./aclImdb/train/" # source data

For the purposes of this project, testing dataset is created from the training dataset after removing polarity.
•	testing_dataset = "./aclImdb/test/testing_dataset.csv" 

Step 2- Algorithms used:
For the purposed of this project unigram, bigram and tfidf algorithms are used to analyze the data.

Step 3- Classifier 
Stochastic Gradient Descent classifier is used in this project to minimize the processing expense instead of gradient descent, as gradient descent tends to be expensive on large dataset. 


Step 4-Analysis on training data:
Accuracy of unigram model is 92.764
Accuracy for the Bigram Model is 93.556
Accuracy for the Unigram TFIDF Model is 88.264
Accuracy for the Bigram TFIDF Model is 86.348


Step 5- Applying the classifier on testing data:
Four files are outputted:
1.	bigram.output.csv
2.	bigramtfidf.output.csv
3.	unigram.output.csv
4.	unigramtfidf.output.csv 


Step 6- Reviewing the results:
Sample from bigram.output.csv, where the polarity has been predicted based on the testing data. 


Step 7: The code file:
Code file:
https://github.com/samirk927/CS410/blob/master/IMDBSentimentAnalysis.py


Performance:
In my testing with 25K training and 25K testing dataset, the code ran for ~75 seconds.



