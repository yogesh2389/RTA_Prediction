# -*- coding: utf-8 -*-
"""
Created on Sun May 14 21:50:28 2023

@author: Yogesh
"""
#import joblib
import numpy as np

def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value


def get_prediction(data, model):
    """
    Predict the class of a given data point.
    """
    return model.predict(data)