'''Bag-of-words + softmax'''
'''only use the imformation A'''
import numpy as np
import pandas as pd
import torch as th
from sklearn.linear_model import LogisticRegression

# load data
fund_labels = [i+1 for i in pd.read_csv('./data/funds_labels.csv')['label'].to_list()]
fund_stocks = [stocks.sum(axis=0) for stocks in np.load('./data/label_mat_list.npy')]
fund_stcoks = []
fundLabel_stocksLabel = [(fundLable, stocksLabel) for fundLable, stocksLabel in zip(fund_labels, fund_stocks)]
fundLabel_stocksLabel

# split train(weight:0.7), val(0.2), test(0.2)
length_data = len(fundLabel_stocksLabel)
length_val = int(len(fundLabel_stocksLabel) * 0.4)
train, val = th.utils.data.random_split(fundLabel_stocksLabel, (length_data - length_val , length_val), generator=th.Generator().manual_seed(42))

train_X = np.array([item[1] for item in train])
train_y = np.array([item[0] for item in train])
# build model
model = clf = LogisticRegression(random_state=0).fit(train_X, train_y)
val_X = np.array([item[1] for item in val])
val_y = np.array([item[0] for item in val])
print(model.score(val_X, val_y))
