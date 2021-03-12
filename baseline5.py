'''baseline5 RNN+BOW+softmax'''
'''use X & A'''

import os
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import json
from tqdm import tqdm
from sklearn import metrics

pre_labels = []
pre_p = []
labels = []

def cal_metrics(model, data):
    with th.no_grad():
        for idx, (fund, stock_labels, fund_label) in enumerate(data):
            model.zero_grad()
            init_hidden = model.init_hidden(fund.shape[1])
            p = model(init_hidden, fund.unsqueeze(2), stock_labels).reshape(1,-1)
            p = th.exp(p)
            pre_p.append(p.data.view(-1).numpy())
            pre_labels.append(np.argmax(p))
            labels.append(fund_label)
    performance = {}
    performance['precision'] = metrics.precision_score(labels, pre_labels, average='weighted')
    performance['acc'] = metrics.accuracy_score(labels, pre_labels)
    performance['recall'] = metrics.recall_score(labels, pre_labels, average='weighted')
    performance['F1'] = metrics.f1_score(labels, pre_labels, average='weighted')
    performance['auroc'] = metrics.roc_auc_score(labels, np.array(pre_p), multi_class='ovo', average='weighted')
    for key, value in performance.items():
        print(key, ":\t", value)
    return performance


''' load data '''
#stock_name - stock_price
with open('./data/stock_prices.json', 'r') as f:
    stock_prices = json.load(f)
    stock_prices = json.loads(stock_prices)  

#fund_labels
fund_labels = pd.read_csv('./data/funds_labels.csv')

#fund - stock_names
DATA_PATH = './data/clean_fund_stock.xlsx'
data = pd.read_excel(DATA_PATH, header=0)
stocks = data.loc[:, [str(i) for i in range(1, 11)]]
assert len(stocks) == len(fund_labels), 'fund_stocks 和 fund_labels 长度不同'
fund_stocks = [stocks for stocks in np.load('./data/label_mat_list.npy')]
funds = list(zip(stocks.values.tolist(), fund_stocks, fund_labels['label']+1))

'''processing data'''

# padding the date: TS_times每一行对应TS_names中的stock的time series
if os.path.exists('./data/temp_baseline4.npy'):
    TS_name_value_dic = np.load('./data/temp_baseline4.npy', allow_pickle=True).item()
    print('use existed temp')
else:
    datetime = [stock['datetime'] for stock in stock_prices]
    stocknames = [stock['name'] for stock in stock_prices]
    all_date_time = set()
    for d in datetime:
        all_date_time = all_date_time | set(d)
    all_date_time = list(all_date_time)
    all_date_time.sort()
    TS_names = stocknames
    datetime = [stock['datetime'] for stock in stock_prices]
    stocknames = [stock['name'] for stock in stock_prices]
    all_date_time = set()
    for d in datetime:
        all_date_time = all_date_time | set(d)
    all_date_time = list(all_date_time)
    all_date_time.sort()
    TS_names = stocknames
    temp = np.zeros((len(stocknames), len(all_date_time)))
    TS_times = pd.DataFrame(temp, columns=list(all_date_time))
    for i in range(len(TS_names)):
        stockname = stocknames[i]
        for stock in stock_prices:
            if stock['name'] == stockname:
                for d, v in zip(stock['datetime'], stock['value']):
                    TS_times.loc[i, str(d)] = v
    TS_name_value_dic = dict(zip(TS_names, np.array(TS_times.values.tolist())))
    np.save("./data/temp_baseline4.npy", TS_name_value_dic)
    print('temp saved!')


dataloader = []

for fund in funds:
    label = fund[2]
    stocks_labels = fund[1]
    stocks = fund[0]
    fund_series = [TS_name_value_dic[str(stock)] for stock in stocks]
    fund_tensor = th.FloatTensor(fund_series).t()
    dataloader.append((fund_tensor, stocks_labels, label))

len_val_test, len_test = int(len(dataloader) * 0.4), int(len(dataloader) * 0.2)
train, val_test = th.utils.data.random_split(dataloader,(len(dataloader)-len_val_test, len_val_test), generator=th.Generator().manual_seed(42))
val, test = th.utils.data.random_split(val_test,(len(val_test)-len_test, len_test), generator=th.Generator().manual_seed(42))

test[1][0].shape # stocks' time series 一列一个
test[1][1].shape # stocks' label 一行一个
test[1][2] # fund label  int

## ''' build model '''

FEAT_SIZE = 1
HIDDEN_SIZE = 64
NUM_CLASSES = len(set(fund_labels['label'])) # 4个
LABEL_SIZE = test[1][1].shape[1] # label维度: 770


import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, label_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size + label_size, num_classes)
        self.softmax = nn.LogSoftmax()  # ???dim
        self.hidden_size = hidden_size
      
    def forward(self, hidden, fund_tensor, stocks_label_tensor):
        # output, hidden = self.lstm(fund_tensor, hidden) 
        output, (hidden, cell) = self.lstm(fund_tensor, hidden)
        # fund :shape (seq_len=431, bsz=10, input_size=1)
        # hidden: (1, bsz=10, hidden_size=128)
        hidden = th.cat([hidden.squeeze(), th.tensor(stocks_label_tensor)], dim=1) # hidden: 10 * ( 128 + 10)
        hidden2 = F.relu(self.linear(hidden).mean(axis=0)) # hidden2: (num_class=4) 
        p = self.softmax(hidden2)
        return p
        
    def init_hidden(self, batch_size, requires_grad=True):
        weight = next(self.parameters())
        # weight.new_zeros(): 拿到一个全0的tensor，类型和weight相同
        # 返回的一个是全0的hidden state, 一个是全0的cell state
        return (weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=requires_grad),
                weight.new_zeros((1, batch_size, self.hidden_size), requires_grad=requires_grad))

model = LSTMModel(FEAT_SIZE, HIDDEN_SIZE, LABEL_SIZE, NUM_CLASSES)



EPOCHS = 100
LEARNING_RATE = 1e-2
optimizer = th.optim.SGD(model.parameters(), lr=LEARNING_RATE)
scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
loss_func = nn.NLLLoss()
val_losses = []


def evaluate_model(model=model, data=val):
    model.eval()
    total_loss = 0.
    with th.no_grad():
        for idx, (fund, stock_labels, fund_label) in enumerate(data):
            model.zero_grad()
            init_hidden = model.init_hidden(fund.shape[1])
            p = model(init_hidden, fund.unsqueeze(2), stock_labels).reshape(1,-1)
            loss = loss_func(p, th.tensor([fund_label]))
            total_loss += loss.item()
    model.train()
    print("eval_loss:", total_loss)
    return total_loss

for epoch in tqdm(range(EPOCHS)):
    model.train()
    epoch_loss = 0
    # hidden, cell = model.init_hidden(dataloader[0][0].shape[1])
    for idx, (fund, stock_labels, fund_label) in enumerate(train):
        model.zero_grad()
        init_hidden = model.init_hidden(fund.shape[1])
        p = model(init_hidden, fund.unsqueeze(2), stock_labels).reshape(1,-1)
        loss = loss_func(p, th.tensor([fund_label]))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    val_loss = evaluate_model(model)
    print('epoch: ', epoch , 'loss: ', epoch_loss)
    if len(val_losses) == 0 or val_loss < min(val_losses):
        print('save best model to baseline5.ph')
        th.save(model.state_dict,'./data/baseline5.ph')
        val_losses.append(val_loss)
    else:
        scheduler.step()
        print('learning_rate decay')
        print('------------')
        print('performance:', cal_metrics(model, test))
    print('---------------')
print('final performance:', cal_metrics(model, test))

