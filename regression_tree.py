#creating a regression tree using cart algorithm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from collections import deque
from copy import deepcopy

df = df_orignal.copy()
df.dropna(axis = 'rows', inplace=True)
y = df['Increase rate']
df = df.drop(['Date'], axis = 1)
df = df.drop(['Increase rate'], axis = 1)


def compute_threshold(feature):
    features_res = []
    thresholds = X_train[feature].unique().tolist()
    thresholds.sort()
    thresholds = thresholds[1:]
    for t in thresholds:
        y_left_ix = X_train[feature] < t
        y_left, y_right = y_train[y_left_ix], y_train[~y_left_ix]
        features_rss.append(rss(y_left, y_right))

    return thresholds, features_res

def selectFeature(features):
    n = len(features)
    return features[np.random.randint(0, n)]

def find_best_rule(X_train, y_train):
    best_feature, best_threshold, min_rss = None, None, np.inf
    
    features = X_train.columns
    #for feature in X_train.columns:
    feature = selectFeature(features)
        #print('checking thresholsds of --> ', feature)
    thresholds = X_train[feature].unique().tolist()
    thresholds.sort()
    thresholds = thresholds[1:]

    for t in thresholds:
        y_left_ix = X_train[feature] < t
        y_left, y_right = y_train[y_left_ix], y_train[~y_left_ix]
        t_rss = rss(y_left, y_right)
        if(t_rss < min_rss):
            min_rss = t_rss
            best_threshold = t
            best_feature = feature
            #print('found bf ', best_feature, ' best th ', t_rss)

    return {'feature':best_feature, 'threshold':best_threshold, 'rmse':np.sqrt(min_rss)}

def split(X_train, y_train, depth, max_depth):
    if(max_depth == -1):
        max_depth = np.inf
    if(depth == max_depth or len(X_train) < 2):
        return {'prediction':np.mean(y_train), 'depth':depth, 'n_samples':len(X_train)}
    
    rule = find_best_rule(X_train, y_train)

    if(rule['rmse'] == 0):
        return {'prediction':np.mean(y_train), 'depth':depth, 'n_samples':len(X_train)}
    # if(rule['rmse'] < 1):
    #     return {'prediction':np.mean(y_train), 'depth':depth}
    #print('SELECTED FEATURE : ', rule['feature'], ' WITH THRESHOLD ', rule['threshold'])
    left_ix = X_train[rule['feature']] < rule['threshold']
    rule['n_samples'] = X_train.shape[0]
    rule['avg'] = np.mean(y_train)
    rule['left'] = split(X_train[left_ix], y_train[left_ix], depth + 1, max_depth)
    rule['right'] = split(X_train[~left_ix], y_train[~left_ix], depth + 1, max_depth)
    rule['depth'] = depth
    return rule

def predict(sample , rules):
    prediction = rules.get('prediction', None)
    while prediction is None:
        feature, threshold = rules['feature'], rules['threshold']
        if(sample[feature] < threshold):
            rules = rules['left']
        else:
            rules = rules['right']
        prediction = rules.get('prediction', None)
    return prediction

def evaluate(X, y, rules):
    preds = X.apply(predict, axis = 'columns', rules = rules.copy())
    return r2_score(y, preds)