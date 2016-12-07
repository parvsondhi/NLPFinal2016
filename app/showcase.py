import pandas as pd
import numpy as np
import string

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from os import path

import pickle

# given a topic for each tweet if a word in topic X exists in the tweet then the feature corresponding is 1, else 0
def topicFeaturesExtract(tweet):
    tweetWords = set(tweet.split())
    topics = ['email', 'russia', 'race', 'immigration', 'trust', 'sex', 'female', 'male']
    output = [0] * len(topics)

    wordLists = {}
    wordLists['email']  = ['emails', 'email', 'crookedhillary', 'prison', 'crooked']
    wordLists['russia'] =  ['putin', 'russia', 'crimea', 'vladimir', 'ukraine', 'russian']
    wordLists['race']   = ['white', 'black', 'racist', 'race']
    wordLists['immigration'] = ['borders', 'border', 'wall', 'mexico', 'illegal', 'immigrants', 'immigration',
                               'trafficking']
    wordLists['trust'] = ['factcheck', 'fact', 'factchecking', 'bigleaguetruth', 'trust', 'politifact',
                          'trustworthiness', 'lies', 'lie', 'truth', 'liar']
    wordLists['sex'] = ['sex', 'sexual', 'assault', 'rape', 'rapist', 'transgression', 'transgressions']

    wordLists['female'] = ['she', 'her', 'she\'s', 'herself']
    wordLists['male'] = ['he', 'his', 'he\'s', 'himself', 'him']

    for idx, val in enumerate(topics):
        if len(tweetWords.intersection(wordLists[val])) > 0:
            output[idx] = 1
        else:
            output[idx] = 0

    return output

# Given a list of tweets return list of lists each sublist is a topic
def topicFeatures(tweetList):
    totalFeatures = [[], [], [], [], [], [], [], []]
    for tweet in tweetList:
        feats = topicFeaturesExtract(tweet)

        for idx, val in enumerate(feats):
            totalFeatures[idx].append(val)

    return totalFeatures

def tweet_processing(tweet_list):
    processed_list = []
    for item in tweet_list:
        item = item.lower().replace('hillary clinton', 'clinton')
        item = item.lower().replace('hrc', 'clinton')
        item = item.lower().replace('bill clinton', 'bill')
        item = item.lower().replace('donald trump', 'trump')
        item = item.lower().replace('tim kaine', 'kaine')
        item = item.lower().replace('mike pence', 'pence')
        item = item.lower().replace('@hillaryclinton', 'clinton')
        item = item.lower().replace('@timkaine', 'kaine')
        item = item.lower().replace('@realDonaldTrump', 'trump')
        item = item.lower().replace('@mike_pence', 'pence')
        processed_list.append(item)
    return processed_list

def trump_occurence(tweet):
    if (('trump' in nltk.word_tokenize(tweet.lower())) or
        ('pence' in nltk.word_tokenize(tweet.lower())) or ('donald' in nltk.word_tokenize(tweet.lower()))):
        return 1
    else:
        return 0

def hillary_occurence(tweet):
    if (('hillary' in nltk.word_tokenize(tweet.lower()))
        or ('kaine' in nltk.word_tokenize(tweet.lower()))
        or ('clinton' in nltk.word_tokenize(tweet.lower()))
        or ('tim' in nltk.word_tokenize(tweet.lower()))):
        return 1
    else:
        return 0

def bill_occurence(tweet):
    if 'bill' in nltk.word_tokenize(tweet.lower()):
        return 1
    else:
        return 0

def dist_hillary(tweet, input_wordList):
    wordlist = nltk.word_tokenize(tweet.lower())
    distances = []

    index = []
    for name in ['hillary','kaine','tim','clinton']:
        try:
            index.append(wordlist.index(name))
        except:
            a=1

    if len(index) == 0:
        return 20

#     if ('hillary' in wordlist):
#         index = wordlist.index('hillary')
#     elif ('clinton' in wordlist):
#         index = wordlist.index('clinton')
#     elif ('kaine' in wordlist):
#         index = wordlist.index('kaine')
#     elif ('tim' in wordlist):
#         index = wordlist.index('tim')
#     else:
#         return 20

    for item in wordlist:
        if item in input_wordList:
            negative_index = wordlist.index(item)
            for idx in index:
                distances.append(abs(negative_index - idx))
    if len(distances)!=0:
        return min(distances)
    else:
        return 20

def dist_trump(tweet, input_wordList):
    wordlist = nltk.word_tokenize(tweet.lower())
    distances = []
#     if ('trump' in wordlist):
#         index = wordlist.index('trump')
#     elif ('donald' in wordlist):
#         index = wordlist.index('donald')
#     elif ('pence' in wordlist):
#         index = wordlist.index('pence')
#     elif ('mike' in wordlist):
#         index = wordlist.index('mike')
#     else:
#         return 20
    index = []
    for name in ['donald','trump','mike','pence']:
        try:
            index.append(wordlist.index(name))
        except:
            a=1

    if len(index) == 0:
        return 20
    for item in wordlist:
        if item in input_wordList:
            negative_index = wordlist.index(item)
            for idx in index:
                distances.append(abs(negative_index - idx))
    if len(distances)!=0:
        return min(distances)
    else:
        return 20

def tokenize_text(corpus):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sents = sent_tokenizer.tokenize(corpus) # Split text into sentences

    output = []
    for sent in raw_sents:
        output = output + nltk.word_tokenize(sent)

    return output

def create_sentences(data_frame):
    mylist = []
    for item in data_frame:
        mylist.append(tokenize_text(item))

    return mylist

def ngram_features_sum(tweet, n=4):
    hillary_noun_list = ['hillary','clinton','kaine', 'tim']
    trump_noun_list   = ['trump', 'donald', 'mike', 'pence']
    features = [[],[]]
    score_list_hillary = 0
    score_list_trump = 0
#     result_set    = create_sentences(tweet)
    sentence_list = create_sentences(tweet)
#     sentence_list = flatten_sentences(result_set)
#     print(len(sentence_list))
    for tokens in sentence_list:
#         print(tokens)
        sum_trump = []
        sum_hillary = []
        for item in tokens:
            index = tokens.index(item)
            if index-n < 0:
                extraEnd = n-index
            else:
                extraEnd = 0

            if index+n > (len(tokens)-1):
                extraStart = index+n - (len(tokens)-1)
            else:
                extraStart = 0

            start = max(0, index-n-extraStart)
            end   = min(len(tokens)-1, index+n+extraEnd)
            n_gram = tokens[start:end+1]
            flat_tweet = ' '.join(n_gram)
            sid    = SentimentIntensityAnalyzer()
            ss = sid.polarity_scores(flat_tweet)
            if item in trump_noun_list:
                sum_trump.append(ss['compound'])
#                 if (index >=4 and index <= len(tokens)-4):
# #                 n_gram = tweet[index-3:index+4]
#                     n_gram = tokens[index-4:index+4]
#                     flat_tweet = ' '.join(n_gram)
# #                     print("tweeting")
# #                     print(flat_tweet)
#                     ss = sid.polarity_scores(flat_tweet)
#                     sum_trump.append(ss['compound'])
#                 elif (index == 0):
#                     n_gram = tokens[index:index+8]
#                     flat_tweet = ' '.join(n_gram)
# #                     print("tweeting")
# #                     print(flat_tweet)
#                     ss = sid.polarity_scores(flat_tweet)
#                     sum_trump.append(ss['compound'])
#                 elif (index == len(tokens)-1):
#                     n_gram = tokens[index-8:index]
#                     flat_tweet = ' '.join(n_gram)
# #                     print("tweeting")
# #                     print(flat_tweet)
#                     ss = sid.polarity_scores(flat_tweet)
#                     sum_trump.append(ss['compound'])
            if item in hillary_noun_list:
                sum_hillary.append(ss['compound'])

#                 if (index >=4 and index <= len(tokens)-4):
#                     n_gram = tokens[index-4:index+4]
#                     #n_gram = tweet[index+0:index+2]
#                     flat_tweet = ' '.join(n_gram)
# #                     print("tweeting")
# #                     print(flat_tweet)
#                     ss = sid.polarity_scores(flat_tweet)
#                     sum_hillary.append(ss['compound'])
#                 elif (index == 0):
#                     n_gram = tokens[index:index+8]
#                     #n_gram = tweet[index+0:index+2]
#                     flat_tweet = ' '.join(n_gram)
# #                     print("tweeting")
# #                     print(flat_tweet)
#                     ss = sid.polarity_scores(flat_tweet)
#                     sum_hillary.append(ss['compound'])
#                 elif (index == len(tokens)-1):
#                     n_gram = tokens[index-8:index]
#                     #n_gram = tweet[index+0:index+2]
#                     flat_tweet = ' '.join(n_gram)
# #                     print("tweeting")
# #                     print(flat_tweet)
#                     ss = sid.polarity_scores(flat_tweet)
#                     sum_hillary.append(ss['compound'])
        if(len(sum_trump) == 0):
            score_list_trump = 0
        else:
            score_list_trump = sum(sum_trump)

        if(len(sum_hillary) == 0):
            score_list_hillary = 0
        else:
            score_list_hillary = sum(sum_hillary)

        #score_list_trump = sum(sum_trump) / float(len(sum_trump))
        #score_list_hillary = sum(sum_hillary) / float(len(sum_hillary))
        features[0].append(score_list_trump)
        features[1].append(score_list_hillary)

    return features

# df is a pandas dataframe with Tweet column
def extractFeatures(df):

    APP_ROOT = path.dirname(path.abspath(__file__))
    df['Tweet'] = tweet_processing(df['Tweet'].tolist())

    # distance feature
    pos_word_list = pickle.load(open(path.join(APP_ROOT,"pos_word_list.pkl"), "rb"))
    neg_word_list = pickle.load(open(path.join(APP_ROOT,"neg_word_list.pkl"), "rb"))

    trump_distance_negative_word   = []
    hillary_distance_negative_word = []
    trump_distance_positive_word   = []
    hillary_distance_positive_word = []
    trump_occured                  = []
    hillary_occured                = []
    bill_occured                   = []

    for tweet in df['Tweet']:
        trump_distance_negative_word.append(dist_trump(tweet, neg_word_list))
        hillary_distance_negative_word.append(dist_hillary(tweet, neg_word_list))
        trump_distance_positive_word.append(dist_trump(tweet, pos_word_list))
        hillary_distance_positive_word.append(dist_hillary(tweet, pos_word_list))
        trump_occured.append(trump_occurence(tweet))
        hillary_occured.append(hillary_occurence(tweet))
        bill_occured.append(bill_occurence(tweet))

    # topic and adjective features
    topicFeats = topicFeatures(df['Tweet'])

    # ngram vader features
    value  = ngram_features_sum(df['Tweet'])

    # compound vader score for entire tweet
    sid    = SentimentIntensityAnalyzer()
    scores = [sid.polarity_scores(tweet)['compound'] for tweet in df['Tweet']]

    # combine the features into an output
    allFeatures = pd.DataFrame()

    allFeatures['trump_distance_negative_word']   = trump_distance_negative_word
    allFeatures['hillary_distance_negative_word'] = hillary_distance_negative_word
    allFeatures['trump_distance_positive_word']   = trump_distance_positive_word
    allFeatures['hillary_distance_positive_word'] = hillary_distance_positive_word
    allFeatures['trump_occured']                  = trump_occured
    allFeatures['hillary_occured']                = hillary_occured
    allFeatures['bill_occured']                   = bill_occured

    topics = ['email', 'russia', 'race', 'immigration', 'trust', 'sex', 'female', 'male']
    for j in range(0,8):
        allFeatures[topics[j]] = topicFeats[j]

    allFeatures['trump_ngram_vader']   = value[0]
    allFeatures['hillary_ngram_vader'] = value[1]

    allFeatures['Score'] = scores
    
    return allFeatures

def RUNME(a_text):
    APP_ROOT = path.dirname(path.abspath(__file__))

    #pipeline = joblib.load("HONESTCLASSIFIER.pkl")
    pipeline = joblib.load(path.join(APP_ROOT,"HONESTCLASSIFIER.pkl"))

    temp = pd.DataFrame()
    temp["Tweet"] = [a_text]

    readableOutput = {1: 'Trump', 0: 'Hillary'}

    return readableOutput[pipeline.predict(extractFeatures(temp))[0]]
