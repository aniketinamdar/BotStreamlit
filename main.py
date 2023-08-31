import streamlit as st
import pandas as pd
import numpy as np
import joblib
import spacy
from transformers import pipeline
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn import metrics as mt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


joblib_in = joblib.load(open("bot_classifier.joblib", "rb"))

nlp = spacy.load('en_core_web_sm')
sentiment_analyzer = pipeline("sentiment-analysis")

def is_subjective(text):
  doc = nlp(text)
  for token in doc:
      if token.dep_ == "nsubj" or token.dep_ == "iobj" or token.dep_ == "dobj":
          return True
  for token in doc:
      if token.pos_ == "PRON" or (token.pos_ == "VERB" and token.dep_ != "aux"):
          return True
  return False


def tweet_checker(tweet):
  tweet = str(tweet)
  tweet = tweet.lower()
  tweet_length = len(tweet)
  sentiment = sentiment_analyzer(tweet)[0]
  sentiment_label = sentiment['label']
  sentiment_score = sentiment['score']
  # print("Tweet length : ",tweet_length)
  # print("Sentiment Label : ",sentiment_label,"\nSentiment Score : ",sentiment_score)

  irony_threshold = 0.2

  if tweet_length <= 70:
    if sentiment_score > irony_threshold:
      return 1
    else:
      if sentiment_label == 'LABEL_1':
        return 0
      elif sentiment_label == 'LABEL_0':
        return 1
  else:
    if is_subjective(tweet):
      return 1
    else:
      return 0
    
def is_old_account(value):
    if value:
        created_date = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        old_threshold = datetime.now() - timedelta(days=365)  
        if created_date <= old_threshold:
            return 0  
        else:
            return 1
    else:
        return 0  # Handle case where timestamp is empty  

def identify_account_age(row):
    if is_old_account(row):
        return 0
    else:
        return 1

def BlankChecker(value):
   if value is not None:
      return 0
   else:
      return 1
   
def StandardChecker(value,number):
  if value:
    if int(value)<=number and int(value)>=0:
      return 0
    else:
      return 1
  else:
     return 1

def BotChecker(input_data):
    scl_obj = StandardScaler()
    new_data_scaled = scl_obj.transform(input_data)
    predicted_prob = joblib_in.predict_proba(new_data_scaled)[0][1]

    if predicted_prob >= 0.8:
        return "Bot ",predicted_prob
    elif predicted_prob >= 0.6:
        return "More likely a bot ",predicted_prob
    elif predicted_prob >= 0.4:
        return "Less likely a bot ",predicted_prob
    else:
        return "Not a bot ",predicted_prob
      
def main():
    st.title('Twitter Bot Detection')
    id = st.text_input("ID", "")

    tweet = st.text_input("Tweet", "")
    tweet_input = tweet_checker(tweet)

    in_reply_to_status_id = st.text_input("Reply to Status id", "")
    in_reply_to_status_id_input = BlankChecker(in_reply_to_status_id)

    in_reply_to_user_id = st.text_input("Reply to User id", "")
    in_reply_to_user_id_input = BlankChecker(in_reply_to_user_id)

    in_reply_to_screen_name = st.text_input("Reply to Screen Name", "")
    in_reply_to_screen_name_input = BlankChecker(in_reply_to_screen_name)
    
    retweeted_status_id = st.text_input("Retweeted Status ID", "")
    retweeted_status_id_input = BlankChecker(retweeted_status_id)

    geo = st.text_input("Geolocation", "")
    geo_input = BlankChecker(geo)

    timestamp = st.text_input("Timestamp","")
    timestamp_input = identify_account_age(timestamp)   

    num_hashtags = st.text_input("Number Hashtags",'')
    num_hashtags_input = StandardChecker(num_hashtags,5)

    num_urls = st.text_input("Number URLs",'')
    num_urls_input = StandardChecker(num_urls,5)

    num_mentions = st.text_input("Number Mentions",'')
    num_mentions_input = StandardChecker(num_mentions,5)

    retweet_count = st.text_input("Number Retweets",'')
    retweet_count_input = StandardChecker(retweet_count,10)

    reply_count = st.text_input("Number Replies",'')
    reply_count_input = StandardChecker(reply_count,10)

    favorite_count = st.text_input("Number Favorites",'')
    favorite_count_input = StandardChecker(favorite_count,10)

    input_data = [[id,
               tweet_input,
               in_reply_to_status_id_input,
               in_reply_to_user_id_input,
               in_reply_to_screen_name_input,
               retweeted_status_id_input,
               geo_input,
               timestamp_input,
               num_hashtags_input,
               num_urls_input,
               num_mentions_input,
               retweet_count_input,
               reply_count_input,
               favorite_count_input
               ]]
      
    if st.button("Bot Checker"):
        output = BotChecker(input_data)
        st.success('Bot or Not : {}'.format(output))


if __name__ == '__main__':
    main()