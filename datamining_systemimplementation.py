# -*- coding: utf-8 -*-
"""
SYSTEM IMPLEMENTATION FOR DETECTING BAD WORDS IN A TEXT

Alliance Mbonigaba
alliancemboni@gmail.com

you can run this files on Colab
    https://colab.research.google.com/drive/1F5TWYZ78wxYMteoLdk2CQgQo5-RAbqAO

# Detecting bad word/s in given text implementation using machine learning
This system will be detecting bad word in given text and replace it with "***"
"""

import time 

# import numpy as np 
# import matplotlib.pyplot as plt

"""Text sentiment """

import nltk
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer 


# calculate sentiment score 
def get_sentiment_score(words):
  #initialize sentiment intenity analyzer object
  sent_obj = SentimentIntensityAnalyzer()
  sentiment_results = sent_obj.polarity_scores(words)
  senti_dataf = pd.DataFrame(columns = ['rate_type', 'percentage'])
  senti_dataf['rate_type'] = ['Negative', 'Neutral', 'Positive', 'overall']
  senti_dataf['percentage'] = [round(sentiment_results['neg'] * 100), round(sentiment_results['neu'] * 100), round(sentiment_results['pos']* 100), round(sentiment_results['compound']* 100)]

  
  #return dataframe with scores
  return senti_dataf['percentage']

#read data
import pandas as pd 

# url = "https://raw.githubusercontent.com/vzhou842/profanity-check/master/profanity_check/data/clean_data.csv"

# dt = pd.read_csv(url)

# dt.head(5)

"""Data Preprocesing"""

# shape
# dt.shape

# check for null values 
# dt.isnull().sum(axis = 0)

# droping null values
# data = dt.dropna(axis=0, how="any", thresh=None, subset=None, inplace=False)

# data.isnull().sum(axis = 0) # check null values

"""Creating and training a model"""

# training model and saving trained models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import joblib

# create a vectorizer to get the count of all stopwords

# vectorizer = CountVectorizer(stop_words='english', min_df=0.0001) #vectorizer

# texts = data['text'].astype(str)

# get x and y axis
# y = data['is_offensive']
# x = vectorizer.fit_transform(texts)

# Train the model
# model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
# model.fit(x, y)

"""Testing the model"""

# test model
# t1 = model.predict(vectorizer.transform(['alliance is just testing the model there should be no bad word']))
# t1 #should return an array [0] since the texts contains no bad words

# t2 = model.predict(vectorizer.transform(['fuck you']))
# t2 # should return an array [1] since the text contains bad word

"""Saving Trained model and vectorizer used for future use"""

# Save trained model
# joblib.dump(vectorizer, 'vectorizer.joblib')
# joblib.dump(model, 'model.joblib')

"""System implementation"""

# get text sentiment before anaysing for bad words
# test the function created 
# test_sentence = 'I have been working so hard, I believe I will get good grades'
# test_get_scr = get_sentiment_score(test_sentence)

# test_get_scr

# check the percent of english used in given text
from nltk import wordpunct_tokenize
from nltk.corpus import words
nltk.download('stopwords')
nltk.download('words')

def detect_english_percent(text):

  #calculate percentage of english in given text
  eng_words_count = 0
  tokens = wordpunct_tokenize(text) #tokenize the text
  for word in tokens:
    if word in words.words():
      eng_words_count = + 1

  return ((eng_words_count / len(tokens)) * 100)

# detect_english_percent(test_sentence)

#label word according to the output from trained model
def lebel_word(word):
  if word == [1]: # the word should be replaced
    return 'bad'
  return 'ok'

#load trained model and vectorizer
loaded_mode = joblib.load('model.joblib')
loaded_vectorizer = joblib.load('vectorizer.joblib')

def predict_bad_word(text):
  array_t = [text]
  p = loaded_mode.predict(loaded_vectorizer.transform(array_t))
 
  return lebel_word(p)

#test prediction
predict_bad_word('This is sucking very bad')

predict_bad_word('stupid')

#replacing the bad word in a given text
def replace_bad(text):
  tokens = text.split()
  
  for i in range(len(tokens)):
    if  predict_bad_word(tokens[i]) == 'bad':
      tokens[i] = '***'
  return tokens

def get_only_profanity(text):
  tokens = text.split()
  profanity = []
  for i in range(len(tokens)):
    if  predict_bad_word(tokens[i]) == 'bad':
      profanity.append(tokens[i])
  return profanity



# get the original text
def get_original(tokens):
  org = ''
  if len(tokens) == 1:
    return tokens[0]
  else:
    org = tokens[0]
    for word in tokens[1:]:
      org = org + ' ' + word
  return org

#test 
tokns = replace_bad('in the city there is  an asshole')
get_original(tokns)

"""The overall function to replace bad words in given text """

#get overall function to replace the badword in text 
def remove_badwords(text):
  tokens = replace_bad(text)
  return get_original(tokens)

#create test cases
# bad1 = 'an asshole is a bad word'
# bad2 = 'I do not enjoy writing a word "fuck" just for the purposes of this assignment'
# good1 = 'There is no bad word'
# good2 = 'just'

# print(remove_badwords(bad1))
# print(remove_badwords(bad2))
# print(remove_badwords(good1))
# print(remove_badwords(good2))