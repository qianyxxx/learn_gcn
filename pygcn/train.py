from __future__ import division
from __future__ import print_function

import time
import numpy as np
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import *
from models import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
# nfeat: 输入特征的维度      --> in_features
# nhid: 隐藏层的维度
# nclass: 输出类别的维度     --> out_features
# dropout: dropout的概率
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)

# Optimizer
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

# 是否使用GPU
if args.cuda:
    # 将模型和数据转移到GPU上
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


# 模型训练
def train(epoch):
    t = time.time()
    # 将模型设置为训练模式
    model.train()
    # 梯度清零
    optimizer.zero_grad()
    # 前向传播
    output = model(features, adj)
    # 计算损失
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # 反向传播
    loss_train.backward()
    # 更新参数
    optimizer.step()

    # 验证集上的损失
    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run
        model.eval()

    # 计算损失
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    # 计算准确率
    acc_val = accuracy(output[idx_val], labels[idx_val])

    # 输出训练过程
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


# 测试模型
def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()

if __name__ == '__main__':
    pass