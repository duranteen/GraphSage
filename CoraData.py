import itertools
import os.path
import pickle
from collections import namedtuple
from scipy import sparse

import numpy as np


class CoraData(object):
    Data = namedtuple('Data', ['x', 'y', 'adjacency', 'train_mask', 'val_mask', 'test_mask'])
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self, data_root="data/cora", rebuild=False):
        """
        加载和处理Cora数据集

        :param data_root: 数据保存目录
        :param rebuild:  是否加载缓存
        """
        self.data_root = data_root
        save_file = os.path.join(self.data_root, "__cached.pkl")
        if os.path.exists(save_file) and not rebuild:
            print("Using Cashed file: {}".format(save_file))
            self._data = pickle.load(open(save_file, "rb"))
        else:
            self._data = self.process_data()
            # with open(save_file, "wb") as f:
            #     pickle.dump(self.data, f)
            # print("Cashed file: {}".format(save_file))

    @property
    def data(self):
        """
        :return: x, y, adjacency, train_mask, val_mask, test_mask
        """
        return self._data


    def process_data(self):
        """
        处理数据，得到节点特征、邻接矩阵，train_mask val_mask test_mask
        :return:
        """
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(os.path.join(self.data_root, name)) for name in self.filenames]
        train_index = np.arange(y.shape[0])
        val_index = np.arange(y.shape[0], y.shape[0] + 500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)

        x[test_index] = x[sorted_test_index]
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)
        val_mask = np.zeros(num_nodes, dtype=np.bool)
        test_mask = np.zeros(num_nodes, dtype=np.bool)
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True
        adjacency = graph
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", len(adjacency))
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return self.Data(x=x, y=y, adjacency=adjacency, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(graph):
        """
        :param graph: 邻接表
        :return: 邻接矩阵
        """
        edge_index = []
        num_nodes = len(graph)
        for src, dst in graph.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        # 去重
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sparse.coo_matrix((np.ones(len(edge_index)),
                                        (edge_index[:, 0], edge_index[:, 1])),
                                      shape=(num_nodes, num_nodes), dtype="float32")
        return adjacency

    @staticmethod
    def read_data(path):
        name = os.path.basename(path)
        if name == "ind.cora.test.index":
            out = np.genfromtxt(path, dtype="int64")
            return out
        else:
            out = pickle.load(open(path, "rb"), encoding="latin1")
            out = out.toarray() if hasattr(out, "toarray") else out
            return out

    @staticmethod
    def normalization(adjacency):
        """
        使用度矩阵归一化邻接矩阵
        计算拉普拉斯矩阵
        L = D^-0.5 * (A + I) * D^-0.5
        :param adjacency:
        :return:
        """
        adjacency += sparse.eye(adjacency.shape[0])
        degree = np.array(adjacency.sum(1))
        d_hat = sparse.diags(np.power(degree, -0.5).flatten())
        L = d_hat.dot(adjacency).dot(d_hat).tocoo()
        return L