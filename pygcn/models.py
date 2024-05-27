import torch
import torch.nn as nn
from layers import GraphConvolution
import torch.nn.functional as F


class GCN(nn.Module):
    """
    定义GCN模型
    nfeat: 输入特征的维度      --> in_features
    nhid: 隐藏层的维度
    nclass: 输出类别的维度     --> out_features
    dropout: dropout的概率
    """

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    # 模型假设输入的图是通过邻接矩阵表示的，x是输入特征，adj是邻接矩阵
    def forward(self, x, adj):
        # 定义GCN的前向传播，首先通过第一个图卷积层，然后使用ReLU激活函数，最后使用dropout进行正则化，防止过拟合
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        # 通过第二个图卷积层，不使用激活函数和dropout
        x = self.gc2(x, adj)
        # 通过log_softmax函数，将输出转换为概率，进行分类，返回预测结果
        return F.log_softmax(x, dim=1)
