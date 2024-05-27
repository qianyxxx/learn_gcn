import math
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):

        super(GraphConvolution, self).__init__()
        # 定义图卷积层的输入特征数和输出特征数
        self.in_features = in_features
        self.out_features = out_features
        # 定义图卷积层的权重参数
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 是否使用偏置
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # 初始化权重参数
        self.reset_parameters()

    def reset_parameters(self):
        # stdv: 标准差
        stdv = 1. / math.sqrt(self.weight.size(1))
        # 使用均匀分布初始化权重参数
        self.weight.data.uniform_(-stdv, stdv)
        # 是否使用偏置
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    # 定义图卷积层的前向传播，其中input为输入特征，adj为邻接矩阵，2708*1433
    def forward(self, input, adj):
        # 将输入特征和权重相乘
        support = torch.mm(input, self.weight)
        # 通过邻接矩阵进行稀疏矩阵乘法
        output = torch.spmm(adj, support)
        # 是否使用偏置
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    # 提供了一个简单的描述，显示了图卷积层的输入特征数和输出特征数
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                + str(self.in_features) + ' -> ' \
                + str(self.out_features) + ')'

