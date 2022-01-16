import numpy as np
import os
import pandas as pd
import torch
import torch.sparse as sparse

from utils.SGCN.common import names, utils


CORRECT_MODEL_FILES_SET = {
    names.MODEL_FILE_NAME,
    names.X_RES_FILE_NAME,
    names.ADJ_RES_NAME,
    names.SH_NAME
}


def get_dataloader_description(
        datasets,
        targets,
        shuffle,
        include_near_native,
        normalize_adj,
        normalize_x):

    descr = ""
    descr += "Data:\n"
    descr += "-- datasets: {}\n".format(datasets)
    descr += "-- targets: {}\n".format(len(targets))
    descr += "-- include_near_native: {}\n".format(include_near_native)
    descr += "-- shuffle: {}\n".format(shuffle)
    descr += "Parameters:\n"
    descr += "-- normalize_adj: {}\n".format(normalize_adj)
    descr += "-- normalize_x: {}\n".format(normalize_x)
    return descr


def get_dataloaders(
        datasets,
        data_path,
        atom_types_path,
        include_near_native,
        normalize_adj,
        normalize_x,
        include_contacts,
        shuffle,
        bad_targets_path=None,
        number=1,
        gpu=0):

    print(' ')
    utils.output('Getting dataloaders...')

    if bad_targets_path is not None:
        bad_targets = utils.get_bad_targets(bad_targets_path)
    else:
        bad_targets = set()
    utils.output('Loaded {} bad targets'.format(len(bad_targets)))

    target_to_path = dict()
    targets = []
    for dataset_name in datasets:
        for target_name, target_path in utils.iterate_targets(utils.path([data_path, dataset_name]), bad_targets):
            target_to_path[target_name] = target_path
            targets.append(target_name)

    targets = sorted(targets)

    residue_type_to_id = utils.get_residue_type_to_id(atom_types_path)
    description = get_dataloader_description(
        datasets=datasets,
        targets=targets,
        shuffle=shuffle,
        include_near_native=include_near_native,
        normalize_adj=normalize_adj,
        normalize_x=normalize_x)
    utils.output(description)

    i = 0
    model_info = [[] for _ in range(number)]
    for target_name in targets:
        for model_name, model_path in utils.iterate_models(
                target_path=target_to_path[target_name],
                include_near_native=include_near_native,
                correct_model_files_set=CORRECT_MODEL_FILES_SET):
            model_info[i % number].append({
                'target_name': target_name,
                'model_name': model_name,
                'model_path': model_path,
            })
            i += 1

    dataloaders = []
    for i in range(number):
        curr_models_info = model_info[i]
        if shuffle:
            np.random.shuffle(curr_models_info)
        loader = DataLoader(
            models_info=curr_models_info,
            residue_type_to_id=residue_type_to_id,
            shuffle=shuffle,
            include_near_native=include_near_native,
            normalize_adj=normalize_adj,
            normalize_x=normalize_x,
            include_contacts=include_contacts,
            description=description,
            gpu=gpu)
        dataloaders.append(loader)

    return dataloaders


class DataLoader:
    def __init__(
            self,
            models_info,
            residue_type_to_id,
            shuffle,
            include_near_native,
            normalize_adj,
            normalize_x,
            include_contacts,
            description,
            gpu):

        self.models_info = models_info
        self.residue_type_to_id = residue_type_to_id
        self.shuffle = shuffle

        self.include_near_native = include_near_native
        self.normalize_x = normalize_x
        self.normalize_adj = normalize_adj
        self.include_contacts = include_contacts
        self.description = description
        self.gpu = gpu

        self.start_pos = 0

    def generate(self, size=None, memsize=512, training=False):
        if size is None:
            curr_models_info = self.models_info
        elif self.start_pos + size < len(self.models_info):
            curr_models_info = self.models_info[self.start_pos:self.start_pos + size]
            self.start_pos += size
        else:
            second_part_size = size - (len(self.models_info) - self.start_pos)
            curr_models_info = self.models_info[self.start_pos:] + self.models_info[:second_part_size]
            self.start_pos = second_part_size

        graphs = []
        for model_info in curr_models_info:
            target_name = model_info['target_name']
            model_name = model_info['model_name']
            model_path = model_info['model_path']
            graph = self.build_graph(model_path=model_path)
            graph.update({
                names.TARGET_NAME_FIELD: target_name,
                names.MODEL_NAME_FIELD: model_name
            })
            if graph['sh'] is None:
                continue
            if graph['y'] is None and training is True:
                continue
            graphs.append(graph)
            if len(graphs) == memsize:
                for g in graphs:
                    yield g
                graphs = []
        for graph in graphs:
            yield graph

    def build_graph(self, model_path):
        one_hot, features = self.load_x(utils.path([model_path, names.X_RES_FILE_NAME]))

        # y can not exist for testing data
        y_path = utils.path([model_path, names.Y_FILE_NAME])
        y = pd.read_csv(y_path, sep=' ')['score'].values if os.path.exists(y_path) else None

        # y_global can not exist for testing data
        y_global_path = utils.path([model_path, names.Y_GLOBAL_FILE_NAME])
        y_global = np.array([float(open(y_global_path).read().strip())]) if os.path.exists(y_global_path) else None

        if self.gpu:
            return {
                names.ONE_HOT_FIELD: torch.from_numpy(one_hot).type(torch.float32).cuda(self.gpu),
                names.FEATURES_FIELD: torch.from_numpy(features).type(torch.float32).cuda(self.gpu),
                names.Y_FIELD: torch.from_numpy(y).type(torch.float32).cuda(self.gpu) if y is not None else None,
                names.Y_GLOBAL_FIELD:
                    torch.from_numpy(y_global).type(torch.float32).cuda(self.gpu) if y_global is not None else None,
                names.SH_FIELD: self.load_sh(
                    utils.path([model_path, names.SH_NAME]),
                    utils.path([model_path, names.ADJ_RES_NAME]),
                    len(one_hot)),
            }
        else:
            return {
                names.ONE_HOT_FIELD: torch.from_numpy(one_hot).type(torch.float32),
                names.FEATURES_FIELD: torch.from_numpy(features).type(torch.float32),
                names.Y_FIELD: torch.from_numpy(y).type(torch.float32) if y is not None else None,
                names.Y_GLOBAL_FIELD: torch.from_numpy(y_global).type(torch.float32) if y_global is not None else None,
                names.SH_FIELD: self.load_sh(
                    utils.path([model_path, names.SH_NAME]),
                    utils.path([model_path, names.ADJ_RES_NAME]),
                    len(one_hot)),
            }

    def load_x(self, file):
        compressed_x = pd.read_csv(file, sep=' ')
        residue_types = compressed_x['residue'].values
        features = compressed_x[['volume', 'buriedness', 'sasa']].values.astype(np.float32)

        if self.normalize_x:
            features_to_normalize = features[:, names.X_NORM_IDX]
            normalizations = 1 / features_to_normalize.sum(axis=0)
            normalized_features = np.einsum('ij,j->ij', features_to_normalize, normalizations)
            features[:, names.X_NORM_IDX] = normalized_features

        one_hot = np.zeros((len(residue_types), len(self.residue_type_to_id)))
        for i, residue_type in enumerate(residue_types):
            one_hot[i, self.residue_type_to_id[residue_type]] = 1

        return one_hot, features

    def load_sh(self, edges_file, adj_file, n):
        if not os.path.exists(edges_file):
            return None
        adj = np.load(edges_file)
        edges = adj.T[0:2].astype(int)
        order = adj.shape[1] - 2
        sh = []

        if self.include_contacts:
            residual_edges = np.genfromtxt(adj_file)
            residual_edges[:, 2] = self.get_normalized_values(residual_edges)
            residual_edges = np.repeat(residual_edges, 2, axis=0)
            coeff = residual_edges[:, 2]
        else:
            coeff = 1

        if self.gpu:
            for i in range(order):
                sh.append(sparse.FloatTensor(torch.LongTensor(edges.T).t(),
                                             torch.FloatTensor(adj.T[2:][i] * coeff),
                                             torch.Size([n, n])).cuda(self.gpu))
        else:
            for i in range(order):
                sh.append(sparse.FloatTensor(torch.LongTensor(edges.T).t(),
                                             torch.FloatTensor(adj.T[2:][i] * coeff),
                                             torch.Size([n, n])))
        return sh

    def get_normalized_values(self, edges):
        if self.normalize_adj:
            sums = np.bincount(edges[:, 0].astype(int), weights=edges[:, 2])
            return np.array([edge[2] / (sums[int(edge[0])] + names.EPSILON) for edge in edges])
        else:
            return edges[:, 2]

    def describe(self):
        return self.description
