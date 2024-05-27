import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    # 获取所有的类别
    classes = set(labels)
    # 创建一个字典，key是类别，value是一个one-hot向量
    class_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # 将标签转换为one-hot向量
    labels_onehot = np.array(list(map(class_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """
    Load citation network dataset (cora only for now)
    :param path:
    :param dataset:
    :return:
    """
    # 打印加载的数据集名称
    print('Loading {} dataset...'.format(dataset))
    # 读取数据集的特征和标签
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    # 将加载的特征信息转换为Scipy的稀疏矩阵格式，并将数据类型转换为np.float32, 2788*1433
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 加载标签信息，将标签转换为one-hot编码的形式
    labels = encode_onehot(idx_features_labels[:, -1])
    # labels = pd.get_dummies(idx_features_labels[:, -1])

    # build graph
    # idx_features_labels[:, 0]表示从所有行中取出第一列，即节点的序号信息
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 创建一部字典idx_map，遍历列表中的每个元素，并为每个元素创建一个键值对，键是原始索引j，值是新索引i，即{原始索引: 新索引}
    idx_map = {j: i for i, j in enumerate(idx)}
    print(idx_map)
    # 加载节点之间的边信息
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # 将边的信息转换为新的索引，首先将edges_unordered数组展平成一维数组，然后遍历每个元素，根据idx_map字典得到新的索引
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    print(edges)
    # 根据节点之间的边信息，创建一个邻接矩阵，即将边信息转换为邻接矩阵的形式，edges.shape[0]表示边的数量, 2708*2708
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    print(adj)

    # build symmetric adjacency matrix
    # T: 转置; multiply: 矩阵对应元素相乘; >: 大于
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 构造稀疏矩阵N，计算一次迭代后的A的总的特征矩阵，cora数据集中的特征矩阵是一个2708*1433的矩阵
    N = sp.csr_matrix((features.shape[0], features.shape[1]), dtype=features.dtype)

    # 遍历邻接矩阵的每一行，即遍历每个节点，cora数据集中有2708个节点
    for i in range(adj.shape[0]):
        # 获取第i个节点的邻居节点的索引，adj[i].indices能识别稀疏矩阵adj的第i行的非零元素的列索引
        neighbor_indices = adj[i].indices

        if len(neighbor_indices) > 0:
            # 获取特征矩阵D中对应的邻居节点的特征
            neighbor_features = features[neighbor_indices]

            # 将邻居节点的特征逐个加到新特征矩阵N的第i行上
            for j, neighbor_index in enumerate(neighbor_indices):
                # 将邻居节点的特征矩阵加到N[i]上，这里可以尝试不同的加法方式
                N[i] += neighbor_features[j]

    # 添加自身的特征
    features = N + features
    print(features)

    # 将特征矩阵和邻接矩阵进行归一化处理
    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # 定义训练集、验证集和测试集的索引范围，这里采用的是固定的索引范围，即前140个节点为训练集，200-500为验证集，500-1500为测试集
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # 将稀疏的特征矩阵先转化为密集的Numpy数组，然后再转化为PyTorch的张量
    features = torch.FloatTensor(np.array(features.todense()))
    # 将标签转化为Numpy数组，并使用np.where(labels)[1]获取每个样本的标签索引，然后转化为PyTorch的长整型张量
    labels = torch.LongTensor(np.where(labels)[1])
    # 将稀疏的邻接矩阵转化为PyTorch的稀疏张量
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # 定义训练集、验证集和测试集的索引范围，然后转化为PyTorch的长整型张量
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # 返回特征矩阵、邻接矩阵、标签、训练集、验证集和测试集
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """
    Row-normalize sparse matrix
    :param mx:
    :return:
    """
    # 计算每行的元素和，并转化为Numpy数组，构造矩阵D，AD(-1)
    rowsum = np.array(mx.sum(1))
    print(rowsum)
    # 计算每一行的倒数，并将其转化为Numpy数组
    r_inv = np.power(rowsum, -1).flatten()
    # 由于某些行的和可能是0,导致倒数为无穷大，因此需要将无穷大的元素设置为0
    r_inv[np.isinf(r_inv)] = 0.
    # 构造对角矩阵，其中对角线元素为每行的倒数
    r_mat_inv = sp.diags(r_inv)
    # 将原始稀疏矩阵乘上对角矩阵，即可得到归一化后的稀疏矩阵
    mx = r_mat_inv.dot(mx)
    return mx


def normalizeDAD(mx):
    """
    Row-normalize sparse matrix
    :param mx:
    :return:
    """
    # 计算每行的元素和，并转化为Numpy数组，构造矩阵D，D（-1/2）* A * D（-1/2）
    rowsum = np.array(mx.sum(1))
    # 计算每一行的倒数，并将其转化为Numpy数组
    r_inv_sqrt = np.power(rowsum, -1 / 2).flatten()
    # 由于某些行的和可能是0,导致倒数为无穷大，因此需要将无穷大的元素设置为0
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    # 构造对角矩阵，其中对角线元素为每行的倒数
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    # 将原始稀疏矩阵乘上对角矩阵，即可得到归一化后的稀疏矩阵
    mx = r_mat_inv_sqrt.dot(mx).dot(r_mat_inv_sqrt)
    return mx


def accuracy(output, labels):
    """
    Compute the accuracy
    :param output:
    :param labels:
    :return:
    """
    # 获取预测的标签
    preds = output.max(1)[1].type_as(labels)
    # 计算预测正确的数量
    correct = preds.eq(labels).double()
    # 计算准确率
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor
    :param sparse_mx:
    :return:
    """
    # 将稀疏矩阵转化为COO格式
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # 创建一个稀疏张量，indices表示非零元素的索引，values表示非零元素的值，size表示稀疏张量的大小
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # 返回稀疏张量
    return torch.sparse_coo_tensor(indices, values, shape)
