import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class Aggregator(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=False, activation=F.relu, aggr_method="mean"):
        """

        :param input_dim: 输入特征维度
        :param out_put_dim: 输出特征维度
        :param use_bias: 是否添加偏置
        :param aggr_method: 邻居聚合的方式
        """
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        # self.activation = activation
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim), requires_grad=True)
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        """
        # 提供均值聚合，加和以及最大池化等聚合操作
        :param neighbor_feature:
        :return:
        """
        if self.aggr_method == "mean":
            aggr_neighbor = neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            aggr_neighbor = neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            aggr_neighbor = neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unknown aggr type, expected sum, max, or mean, but got {}"
                             .format(self.aggr_method))

        neighbor_hidden = torch.matmul(aggr_neighbor, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias
        # if self.avtivation:
        #     neighbor_hidden = self.activation(neighbor_hidden)
        return neighbor_hidden

    def extra_repr(self):
        return 'in_features={}, out_features={}, aggr_method={}'.format(
            self.input_dim, self.output_dim, self.aggr_method)


class AggreGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 activation=F.relu,
                 aggr_neighbor_method="mean",
                 aggr_hidden_method="sum"):
        """
        完成目标节点与邻居节点嵌入特征聚合
        :param input_dim:
        :param hidden_dim:
        :param activation: 激活函数 relu
        :param aggr_neighbor_method: 聚合邻域
        :param aggr_hidden_method: 聚合目标节点与邻居特征的方式：sum or 拼接
        """
        super(AggreGCN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.aggr_neighbor_method = aggr_neighbor_method
        self.aggr_hidden_method = aggr_hidden_method
        self.aggregator = Aggregator(input_dim, hidden_dim, aggr_method=aggr_neighbor_method)
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_features, neighbor_features):
        neighbor_hidden = self.aggregator(neighbor_features)
        self_hidden = torch.matmul(src_features, self.weight)
        if self.aggr_hidden_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hidden_method == "cat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, but got {}"
                             .format(self.aggr_hidden_method))

        if self.activation:
            hidden = self.activation(hidden)
        return hidden


class GraphSage(nn.Module):
    def __init__(self, input_dim, hidden_dim=[64, 64], num_neighbor_list=[10, 10]):
        """
        :param input_dim:
        :param hidden_dim: 每个隐藏层的维度大小
        :param num_neighbor_list: [10, 10] 两层网络，每层采样10个邻居，
        """
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbor_list = num_neighbor_list
        self.num_layers = len(num_neighbor_list)
        self.gcn = nn.ModuleList()
        self.gcn.append(AggreGCN(input_dim, hidden_dim[0]))
        for index in range(0, len(hidden_dim) - 2):
            self.gcn.append(AggreGCN(hidden_dim[index], hidden_dim[index+1]))
        self.gcn.append(AggreGCN(hidden_dim[-2], hidden_dim[-1], activation=None))

    def forward(self, node_features_list):
        """
        :param node_features_list: 节点的特征列表，第0个为目标节点，其余为邻居节点
        :return:
        """
        hidden = node_features_list
        # print(len(hidden))
        for layer in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[layer]
            for hop in range(self.num_layers - layer):
                src_feat = hidden[hop]
                src_feat_len = len(src_feat)
                neighbor_node_feat = hidden[hop+1].view(
                    (src_feat_len, self.num_neighbor_list[hop], -1)
                )
                h = gcn(src_feat, neighbor_node_feat)
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]
