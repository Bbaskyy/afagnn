import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import scipy.io
import numpy as np


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree



class AFALayer(MessagePassing):
    def __init__(self, data, num_hidden, dropout):
        super(FALayer, self).__init__(aggr='add')
        self.data = data
        self.dropout = nn.Dropout(dropout)
        self.num_hidden = num_hidden
        self.gate = nn.Linear(2 * num_hidden, num_hidden)
        self.row, self.col = data.edge_index
        self.norm_degree = degree(self.row, num_nodes=data.y.shape[0]).clamp(min=1)   #计算节点的度 输出一个一维tensor表示每个节点的度
        self.norm_degree = torch.pow(self.norm_degree, -0.5)  #相当于 D^(-1/2)  用于GCN的
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        self.g = None  # 初始化中间变量g为None


    def forward(self, h):    # h是 data.x
        h2 = torch.cat([h[self.row], h[self.col]], dim=1)  #aG=tanh(gt[hi||hj])
        # g = torch.tanh(self.gate(h2)).squeeze()  #aG=tanh(gt[hi||hj])
        self.g = torch.tanh(self.gate(h2))
        self.g = torch.full_like(self.g, -1)
        # g = self.gate(h2)
        g1 = self.norm_degree[self.row] * self.norm_degree[self.col]
        g1 = g1.expand((self.num_hidden, 1)).T
        # norm = g * self.norm_degree[self.row] * self.norm_degree[self.col]  #  aG/(didj)^0.5
        norm = self.g * g1
        norm = self.dropout(norm)
        return self.propagate(self.data.edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)

    def message(self, x_j, norm):
        return norm * x_j

    def update(self, aggr_out):
        return aggr_out


class AFAGNN(nn.Module):
    def __init__(self, data, num_features, num_hidden, num_classes, dropout, eps, layer_num=2):
        super(FAGCN, self).__init__()

        self.eps0 = eps
        self.eps1 = torch.nn.Parameter(torch.rand(1, num_hidden), requires_grad=True)

        self.layer_num = layer_num
        self.dropout = dropout
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(AFALayer(data, num_hidden, dropout))  #这里选模型
        self.t1 = nn.Linear(num_features, num_hidden)
        self.t2 = nn.Linear(num_hidden, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h)
            h = (torch.sigmoid(self.eps1)+1) * raw + h
        fea = h
        h = self.t2(h)
        return F.log_softmax(h, 1),fea


    def save_g_to_mat(self, path):
        g_values = {}
        for i, layer in enumerate(self.layers):
            if layer.g is not None:
                g_values[f'layer_{i}_g'] = layer.g.cpu().detach().numpy()

        # 保存每一层的中间变量g为.mat文件
        scipy.io.savemat(path, g_values)

    def g_to_mat(self, path):
        g_values = {}
        for i, layer in enumerate(self.layers):
            if layer.g is not None:
                g_values[f'layer_{i}_g'] = layer.g.cpu().detach().numpy()

        return g_values[f'layer_{i}_g']


    def fea(self):
        return self.fea
