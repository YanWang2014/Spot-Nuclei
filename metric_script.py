#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:53:52 2018

@author: wayne

calculate metric based on two csvs (which is better for offline validation) or two instance aware mask np arrays 
    like what we do in metric_test, which is better for online validation
"""

import pandas as pd

target = pd.read_csv('data/stage1_train_labels.csv')
source = pd.read_csv('results/UnetVgg11_val.csv')


def submit_metric(source, target):
    # https://www.kaggle.com/c/data-science-bowl-2018#evaluation
    
    return 0




print(submit_metric(target, target))
print(submit_metric(source, target))
