import numpy as np

def sampling(src_nodes, sample_num, neighbor_table):
    """

    :param src_nodes: 源节点
    :param sample_num: 采样指定数量的邻居节点，有放回的采样，防止节点邻居数量不足够采样
    :param neighbor_table: 节点与邻居连接表
    :return:
    """
    results = []
    for src in src_nodes:
        res = np.random.choice(neighbor_table[src], size=(sample_num,))
        results.append(res)
    return np.asarray(results).flatten()


def multi_hop_sampling(src_nodes, sample_num, neighbor_table):
    """
    多级采样
    :param src_nodes: 源节点
    :param sample_num: int 采样指定数量的邻居节点，有放回的采样，防止节点邻居数量不足够采样
    :param neighbor_table: 节点与邻居连接表
    :return:
    """
    sampling_result = [src_nodes]
    for k, hopk_num in enumerate(sample_num):
        hopk_result = sampling(sampling_result[k], hopk_num, neighbor_table)
        sampling_result.append(hopk_result)
    return sampling_result