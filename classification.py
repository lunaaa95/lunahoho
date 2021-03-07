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

USE_CUDA = th.cuda.is_available()
print(USE_CUDA)

def process_prec_mat(precision_matrix, atol):
    # 大于atol设1，小于设0，对角线全0
    length = len(precision_matrix)
    precision_matrix[precision_matrix > atol] = 1
    precision_matrix[precision_matrix <= atol] = 0
    precision_matrix[range(length), range(length)] = 0
    precision_matrix = precision_matrix
    return precision_matrix

def to_dgl_graph(mat, label):
    length = len(mat)
    edges = []
    for i in range(length):
        for j in range(i, length):
            if abs(mat[i,j]-1) <= 1e-6:
                edges.append((i,j))

    g = dgl.graph(([edge[0] for edge in edges], [edge[1] for edge in edges]), num_nodes=10)
    g = dgl.to_bidirected(g)
    g.ndata['h'] = th.FloatTensor(label)
    return g


# load tensor_list: [10 * 10的ndarray, ...]
tensor_list = np.load('./data/X.npy')
# tensors: K * p * p : [fund1,...] 图结构信息
tensors = tensor_list.copy()
tensors = [process_prec_mat(tens, 0.04) for tens in tensors]
# _ = show_tensor(tensors[20:30])

# load label_mat_list:[fund_label:[stock_label * 10], ...]
label_mat_list = np.load('./data/label_mat_list.npy', allow_pickle=True)
assert(len(label_mat_list) == len(tensor_list)), 'num_label and num_fund should be the same'

# load fund_label
fund_label_list = pd.read_csv('./data/funds_labels.csv')['label'].tolist()



# pict a graph
G = nx.Graph()
G.add_nodes_from([i for i in range(10)])
tensor = tensors[1]
for i in range(10):
    for j in range(i, 10):
        if abs(tensor[i, j]-1) < 1e-6:
            G.add_edge(i, j)
nx.draw(G)



'''split dataset'''

graphs = [(to_dgl_graph(tensor, label), fund_label+1) for tensor, label, fund_label in zip(tensors, label_mat_list, fund_label_list)]
val_test_data, train_data = th.utils.data.random_split(graphs, [int(len(graphs) * 0.4), len(graphs) - int(len(graphs) * 0.4)], generator=th.Generator().manual_seed(42))
val_data, test_data = th.utils.data.random_split(val_test_data, [int(len(val_test_data) * 0.5), len(val_test_data) - int(len(val_test_data) * 0.5)], generator=th.Generator().manual_seed(42))

# content is graph, dataloader怎么做



class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.linear = nn.Linear(h_feats, num_classes, bias=True)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, g, in_feat):
        # in_feat: 10 * 16
        h = self.conv(g, in_feat) # h: (10 * h_feats)
        h = F.relu(h)
        aggregation = h.mean(axis = 0) # aggregation: (1 * h_feats)
        h2 = self.linear(aggregation) # h2: (1 * num_classes)
        p = self.softmax(h2)# p: (1 * num_classes)
        return p

def evaluate(model, data=val_data):
    model.eval()
    total_loss = 0.
    with th.no_grad():
        for batch_idx, (graph, fund_label) in enumerate(data):
            p = model(graph, graph.ndata['h']).reshape(1,-1)
            loss = loss_func(p, th.tensor([fund_label]))
            total_loss += loss.item()
    model.train()
    print("eval_loss:", total_loss)
    return total_loss

from sklearn import metrics
def cal_precision(model, data=test_data):
    pre_labels = []
    pre_p = []
    labels = []
    total_cnt = len(data)
    correct_cnt = 0
    with th.no_grad():
        for idx, (graph, fund_label) in enumerate(data):
            p = model(graph, graph.ndata['h']).reshape(1,-1)
            pre_p.append( p.data.view(-1).numpy())
            pre_labels.append(np.argmax(p))
            labels.append(fund_label)
    for i in range(total_cnt):
        if pre_labels[i] == labels[i]:
            correct_cnt += 1
    performance = {}
    performance['precision'] = metrics.precision_score(labels, pre_labels, average='weighted')
    performance['acc'] = metrics.accuracy_score(labels, pre_labels)
    performance['recall'] = metrics.recall_score(labels, pre_labels, average='weighted')
    performance['F1'] = metrics.f1_score(labels, pre_labels, average='weighted')
    print('pre_p.shape', np.array(pre_p))
    performance['auroc'] = metrics.roc_auc_score(labels, np.array(pre_p), multi_class='ovo', average='weighted')
    for key, value in performance.items():
        print(key, ":\t", value)
    return performance

# create a GCN model

H = 16
NUM_CLASSES = 9

g1 = to_dgl_graph(tensors[1], label_mat_list[1])
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
    val_loss = evaluate(model)
    if len(val_losses) == 0 or val_loss < min(val_losses):
        print('save best model to model.ph')
        th.save(model.state_dict,'./data/model.ph')
        val_losses.append(val_loss)
    else:
        scheduler.step()
        print('learning_rate decay')
        print('performance:', cal_precision(model))
    print('---------------')
print('final performance:', cal_precision(model))


