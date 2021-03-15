# 单位阵做邻接矩阵+图分类
'''prepare graphs'''
import numpy as np
import pandas as pd
import networkx as nx
import torch as th
import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import matplotlib.pyplot as plt
from explib._helper import *
from main_model import *

USE_CUDA = th.cuda.is_available()
print(USE_CUDA)

def to_zeros_dgl_graph(label):
    length = len(label)
    g = dgl.graph(([i for i in range(length)], [i for i in range(length)]), num_nodes=length)
    g.ndata['h'] = th.FloatTensor(label)
    return g

# load label_mat_list:[fund:[stock_label * 10], ...]
label_mat_list = np.load('./data/label_mat_list.npy', allow_pickle=True)

# load fund_label:[fund_label,...]
fund_label_list = pd.read_csv('./data/funds_labels.csv')['label'].tolist()

'''split dataset'''
graphs = [(to_zeros_dgl_graph(label), fund_label + 1) for label, fund_label in zip(label_mat_list, fund_label_list)]
val_test_data, train_data = th.utils.data.random_split(graphs, [int(len(graphs) * 0.4), len(graphs) - int(len(graphs) * 0.4)], generator=th.Generator().manual_seed(42))
val_data, test_data = th.utils.data.random_split(val_test_data, [int(len(val_test_data) * 0.5), len(val_test_data) - int(len(val_test_data) * 0.5)], generator=th.Generator().manual_seed(42))

# content is graph, dataloader怎么做

from sklearn import metrics

# create a GCN model

H = 16
NUM_CLASSES = 4

g1 = to_zeros_dgl_graph(label_mat_list[1])
g1

model = GCN(g1.ndata['h'].shape[1], H, NUM_CLASSES)

EPOCHS = 30
LEARNING_RATE = 1e-2
optimizer = th.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
loss_func = nn.CrossEntropyLoss()
val_losses = []
for epoch in range(EPOCHS): 
    model.train()
    epoch_loss = 0
    for batch_idx, (graph, fund_label) in enumerate(train_data):
        optimizer.zero_grad()
        # graph.ndata['h'].shape: 10 * 770
        p = model(graph, graph.ndata['h']).reshape(1,-1)
        loss = loss_func(p, th.tensor([fund_label]))
        loss.backward(retain_graph=True)
        optimizer.step()
        epoch_loss += loss.detach().item()
    print('epoch loss:', epoch_loss)
    val_loss = evaluate(model, val_data, loss_func)
    if len(val_losses) == 0 or val_loss < min(val_losses):
        print('save best model to baseline2.ph')
        th.save(model.state_dict,'./data/baseline2.ph')
        val_losses.append(val_loss)
    else:
        scheduler.step()
        print('learning_rate decay')
        print('performance:', cal_precision(model, test_data, loss_func))
    print('---------------')
print('final performance:', cal_precision(model, test_data, loss_func))


