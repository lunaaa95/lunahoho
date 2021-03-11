import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import json
from tqdm import tqdm, trange

'''one hot encoding'''
DATA_PATH = './data/clean_fund_stock.xlsx'
# load data
data = pd.read_excel(DATA_PATH, header=0)
stocks = (data.loc[:, [str(i) for i in range(1, 11)]]).values
# flat data
all_stocks = []
for row in stocks:
    all_stocks.extend(row)

'''
# output stocks' names
names = [str(i) for  i in set(all_stocks)]
names[0:3]
with open('./data/stocks_names.txt', 'w') as f:
    f.write('\n'.join(names))
'''
all_stocks = np.array(all_stocks).reshape(-1,1)
# encode data
onehot_encoder = OneHotEncoder(sparse=False ,dtype=int, handle_unknown = "ignore")
onehot_encoded = onehot_encoder.fit_transform(all_stocks)
VOCAB_NUM = onehot_encoded.shape[1]
# input a int, output a stock_name_string
def idx_to_name(idx, onehot_encoder=onehot_encoder, vacab_num=VOCAB_NUM):
    assert(idx < vacab_num), "one-hot-array's idx should less than vacab_num"
    x = np.zeros((vacab_num), dtype=int).reshape(1, -1)
    x[0, idx] = 1
    return onehot_encoder.inverse_transform(x)

# input: list:['06969.HK','000001.SZ'], output a one-hot-array
def name_to_array(str_name_list, onehot_encoder=onehot_encoder):
    arr = np.array(str_name_list).reshape(-1, 1)
    return onehot_encoder.transform(arr)

def name_to_idx(str_name_list, onehot_encoder=onehot_encoder):
    arr_list = name_to_array(str_name_list, onehot_encoder=onehot_encoder)
    return [np.argmax(arr) for arr in arr_list]
'''calculate covariance'''
def match_series(stock1, stock2):
    date_in_common = list(set(stock1['datetime']) & set(stock2['datetime']))
    date_in_common.sort(reverse=False)
    dict1 = dict(zip(stock1['datetime'], stock1['value']))
    dict2 = dict(zip(stock2['datetime'], stock2['value']))
    return [dict1[i] for i in date_in_common], [dict2[i] for i in date_in_common]

def de_mean(x):
    xmean = np.mean(x)
    return [xi - xmean for xi in x]

def covariance(x1, x2):
    n = len(x1)
    cov = abs(np.dot(de_mean(x1), de_mean(x2)) / n-1) if n>1 else 0
    return cov

with open('./data/stock_prices.json', 'r') as f:
    stock_prices = json.load(f)
    stock_prices = json.loads(stock_prices)
# stock_prices: [{'name':string, 'datetime':['20180830',...], 'value':[7.68,...], 'status:[-1,...]'},{}]d

stock_idx = name_to_idx([stock['name'] for stock in stock_prices])
temp = list(zip(stock_prices, stock_idx))
temp.sort(key=lambda x: x[1])
# 排序 cov_matrix 中x[i, j]表示stock_names中idx=i 和 idx=j 的stocks的cov
stock_prices = [i[0] for i in temp]
stock_names = [stock['name'] for stock in stock_prices]


if os.path.exists('./data/cov_mat.txt'):
    cov_matrix = np.loadtxt('./data/cov_mat.txt')
    print('use prepared cov_mat')
else:
    cov_matrix = np.zeros((VOCAB_NUM, VOCAB_NUM), dtype=float)
    for i in tqdm(range(VOCAB_NUM-1)):
        for j in range(i, VOCAB_NUM):
            matched_val_1, matched_val_2 = match_series(stock_prices[i], stock_prices[j])
            cov_matrix[i,j] = covariance(matched_val_1, matched_val_2)
            cov_matrix[j,i] = cov_matrix[i,j]
    np.savetxt('./data/cov_mat.txt', cov_matrix)
    print('use new cov_mat and success save it')
#test = np.loadtxt('cov_mat.txt')

from explib.models.mgl_opt import solve_mgl

label_mat_list = [np.array(name_to_array(i)) for i in stocks]
print('label_mat_list prepared!')

emp_cov_list = []
for fund in tqdm(stocks):
    length = len(fund)
    idx = name_to_idx(fund)
    cov_x = np.zeros((length, length))
    for i in range(length):
        for j in range(i,length):
            cov_x[i][j] = cov_matrix[idx[i],idx[j]]
            cov_x[j][i] = cov_x[i][j]
    emp_cov_list.append(cov_x)
print('emp_cov_list prepared!')

seed = 0
with np.errstate(all='raise'):
# emp_cov_list is the sample covariance matrix.
# Ak is the i-th variable’s attributes in the k-th task
    X_list, U, fvals = solve_mgl(emp_cov_list, label_mat_list,
                                 d=100, 
                                 lambda_1=.2,
                                 lambda_2=.04,
                                 beta=.01,
                                 outer_max_it=20,
                                 outer_tol=1e-4,
                                 seed=seed)

from explib._helper import *
print('cal success!')

X_filtered_list = [X.copy() for X in X_list]
for X in X_filtered_list:
    X[np.isclose(X_list[0], 0, atol=.04)] = 0
    pass
print('------')
np.save("Y.npy", X_filtered_list)
print('Y_filetered_list saved!')
# show_tensor(X_filtered_list)
