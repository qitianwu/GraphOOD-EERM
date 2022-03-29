import scipy.io
import numpy as np
import scipy.sparse
import torch
import csv
import json
from os import path
import pickle as pkl

def get_id(old_id, time_step, time_step_cnt, new_id):
    if new_id.get(old_id) is None:
        if time_step_cnt.get(time_step) is None:
            time_step_cnt[time_step] = 0
        new_id[old_id] = time_step_cnt[time_step]
        time_step_cnt[time_step] += 1
    return new_id[old_id]


def load_elliptic(data_dir):
    feature_list = open(f'{data_dir}/elliptic/elliptic_txs_features.csv').readlines()
    edge_list = open(f'{data_dir}/elliptic/elliptic_txs_edgelist.csv').readlines()[1:]  # first line is useless
    label_list = open(f'{data_dir}/elliptic/elliptic_txs_classes.csv').readlines()[1:]  # first line is useless

    time_step_cnt = {}
    new_id = {}
    time_steps = {}
    result = []

    for line in feature_list:
        features = line.strip().split(',')
        old_id = int(features[0])
        time_step = int(features[1]) - 1  # 1-base to 0-base
        time_steps[old_id] = time_step
        features = list(map(float, features[2: ]))

        while len(result) <= time_step:
            result.append([None, [], []])

        node_id = get_id(old_id, time_step, time_step_cnt, new_id)

        while len(result[time_step][2]) <= node_id:
            result[time_step][2].append(None)

        result[time_step][2][node_id] = features

    for line in label_list:
        label = line.strip().split(',')
        old_id = int(label[0])
        label = label[1]

        if label == 'unknown':
            label = -1
        else:
            label = 0 if int(label) == 2 else 1

        time_step = time_steps[old_id]
        node_id = get_id(old_id, time_step, time_step_cnt, new_id)

        while len(result[time_step][1]) <= node_id:
            result[time_step][1].append(None)
        result[time_step][1][node_id] = label

    for i in range(len(result)):
        result[i][0] = []
    #     result[i][0] = np.zeros((time_step_cnt[i], time_step_cnt[i]))

    for edge in edge_list:
        u, v = edge.strip().split(',')
        u = int(u)
        v = int(v)
        time_step = time_steps[u]

        u = get_id(u, time_step, time_steps, new_id)
        v = get_id(v, time_step, time_steps, new_id)

        result[time_step][0].append([u, v])

        # result[time_step][0][u][v] = 1

    # for i in range(len(result)):
    #     A = result[i][0]
    #     num = 0
    #     for k in range(A.shape[0]):
    #         num += ( np.sum(A[k] > 0) >= 2 )
    #     print(num, A.shape[0])

    for i in range(len(result)):
        edge_list = result[i][0]
        src, targ = zip(*edge_list)
        A = scipy.sparse.csr_matrix((np.ones(len(src)),
                                     (np.array(src), np.array(targ))),
                                    shape=(time_step_cnt[i], time_step_cnt[i]))
        result[i][0] = A
        result[i][1] = np.array(result[i][1]).astype(np.int)
        result[i][2] = np.array(result[i][2]).astype(np.float32)

        with open(file_path + '/{}.pkl'.format(i), 'wb') as f:
            pkl.dump(result[i], f, pkl.HIGHEST_PROTOCOL)

    #     print(len(src), time_step_cnt[i])


    # print(result[0][1].shape)
    # print(result[0][2].shape)
    # print(len(result[0][1]), len(result[0][2]))

    return result



if __name__ == '__main__':
    load_elliptic()


