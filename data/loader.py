import os
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import sparse


def expand_sh(adj, n, gpu=False, np_format=True):
    edges = adj.T[0:2].astype(int)
    order = adj.shape[1] - 2
    sh = []
    coeff = 1
    if gpu:
        for i in range(order):
            sh.append(sparse.FloatTensor(torch.LongTensor(edges.T).t(),
                                         torch.FloatTensor(adj.T[2:][i] * coeff),
                                         torch.Size([n, n])).cuda(gpu))
    else:
        for i in range(order):
            sh.append(sparse.FloatTensor(torch.LongTensor(edges.T).t(),
                                         torch.FloatTensor(adj.T[2:][i] * coeff),
                                         torch.Size([n, n])))
    sh = torch.stack([i.to_dense() for i in sh])
    if np_format:
        sh = sh.numpy().astype(np.float32)
    return sh


class data_generator(Dataset):

    def __init__(self, model_data_dir, target_data_dir, disto_type='base', target_list=None):
        self.model_data_dir = model_data_dir
        self.target_data_dir = target_data_dir
        self.disto_type = disto_type
        if target_list is None:
            self.data_list = [f for f in os.listdir(self.model_data_dir) if re.match(r'.+.npz', f)]
        else:
            self.data_list = [f for f in os.listdir(self.model_data_dir) if re.match(r'.+.npz', f) and f.split('.')[0] in target_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        model_data_file = os.path.join(self.model_data_dir, self.data_list[idx])
        target_name = self.data_list[idx].split('.')[0]
        x = np.load(model_data_file)
        one_hot = x['one_hot']
        features = x['features']
        pos = x['pos']
        sh = x['sh']
        el_src = x['el_src']
        el_dst = x['el_dst']
        diff_bins = x['diff_bins']
        pos_label_superpose = x['pos_label_superpose']
        lddt_label = x['lddt_label']
        sh_data = expand_sh(sh, pos.shape[0], np_format=False)
        if self.disto_type == 'base':
            disto_feature = x['disto_feature']
            f1d = np.concatenate((one_hot, features), axis=0)
            f2d = np.concatenate((sh_data, disto_feature), axis=0)
            return f1d, f2d, pos, (el_src, el_dst), diff_bins, pos_label_superpose, lddt_label
        else:
            target_data_file = os.path.join(self.target_data_dir, target_name + '.npz')
            af2_qa = x['af2_qa']
            x_target = np.load(target_data_file)
            af2_2d = x_target['disto_feature']
            plddt = x_target['plddt']
            cmap = x_target['cmap']
            f1d = np.concatenate((one_hot, features, plddt, af2_qa), axis=0)
            f2d = np.concatenate((sh_data, af2_2d), axis=0)
            return f1d, f2d, pos, (el_src, el_dst), cmap, diff_bins, pos_label_superpose, lddt_label
