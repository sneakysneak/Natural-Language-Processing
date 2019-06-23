#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 14:06:33 2019

@author: sneakysneak
"""

# Natural LAnguage Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
# dataset['Review'][0]
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Need a new list, populate it with all the reviews
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # 2nd step of cleaning process, update review variable
    # make chars low
    review = review.lower()
    # remove non significant words; on, the, a, an; Use nltk stopwords
    review = review.split()
    ps = PorterStemmer()
    # Iterate through on all words if NOT the word in stopword's list 
    # Stopwords nltk package must be importated
    # Put in set, Sets are faster than lists in Python
    # Stem the words - call the library so ps.
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Final step is joining back diff words like - wow, love, place -> wow love place so 1 string
    review = ' '.join(review)
    # append the cleaned review to the corpus
    corpus.append(review)
 # Create a Bag of Words model   
from sklearn.feature_extraction.text import CountVectorizer
# Creating an object - call the CountVectorizer class
# Max features - filter the non relevant words from 1565 to 1500
cv = CountVectorizer(max_features = 1500)
# X is the matrix of features
X = cv.fit_transform(corpus).toarray()
# take the index of the dependend variable vector - column
# SO we need the results to
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(74+68)/200