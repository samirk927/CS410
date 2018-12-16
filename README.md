# CS410
IMDB Review Sentiment Analysis

IMDB or Internet Movie Database provides diverse information on Movies, TV series and their associated cast, crew, biographies, trivia etc. One of the most interesting and important information that it provides is reviews and ratings. The objective of this project is to train the model and predict the sentiment on the testing dataset.

Dataset for this project has been gather from:
Source:
http://ai.stanford.edu/~amaas/data/sentiment/

The dataset contains 50,000 reviews from IMDB which are divided evenly into 25K training and 25K testing dataset. For the purpose of this project the testing dataset has been created by removing the polarity from the training dataset. On the entire collection, no more than 30 reviews are allowed per movie. Any movie with a rating score of <=4 is considered as negative and >=7 as positive. Neutral ratings(rating 5 and 6) are not considered.

Files

There are two main directories (train/ and test/) which corresponds to training and testing dataset. They include both positive and negative reviews. 
