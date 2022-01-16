import datetime
import numpy as np
import os
import re
import shutil

from utils.SGCN.common import names


def path(parts):
    return os.path.join(*parts)


def mkdir(p, overwrite=False):
    if os.path.exists(p) and overwrite:
        shutil.rmtree(p)
    if not os.path.exists(p):
        os.makedirs(p)


def parse_datasets(arg):
    if ',' in arg:
        casp_folders = arg.split(',')
    else:
        casp_folders = [arg]
    if any([re.match('CASP[0-9]+', s) is None for s in casp_folders]):
        raise Exception('Incorrect CASP numbers format: {}'.format(arg))
    return casp_folders


def get_experiment_description_light(train_args):
    return [
        'id: {}'.format(train_args.id),
        'features: {}'.format(train_args.features),
        'network: {}'.format(train_args.network),
        'conv_nonlinearity: {}'.format(train_args.conv_nonlinearity),
        '-',
        'train_datasets: {}'.format(train_args.train_datasets),
        'include_near_native: {}'.format(train_args.include_near_native),
        'normalize_adj: {}'.format(train_args.normalize_adj),
        'normalize_x: {}'.format(train_args.normalize_x),
        'global_normalization: {}'.format(train_args.global_normalization),
        'res_seq_sep: {}'.format(train_args.res_seq_sep),
        'shuffle: {}'.format(train_args.shuffle),
        'threads: {}'.format(train_args.threads),
        'bad_targets: {}'.format(train_args.bad_targets is not None),
        '-',
        'optim: {}'.format(train_args.optim),
        'lr: {}'.format(train_args.lr),
        'loss: {}'.format(train_args.loss),
        'train_size: {}'.format(train_args.train_size)
    ]


def parse_experiment_description(description):
    return dict(map(lambda p: tuple(p.split(': ')), filter(lambda s: s != '-', description.split('\n'))))


def get_checkpoint(checkpoint_path, model_id):
    if checkpoint_path is not None:
        checkpoint_path = path([checkpoint_path, model_id])
        if not os.path.exists(checkpoint_path):
            print(' ')
            output('Directory {} does not exist yet'.format(checkpoint_path))
            os.makedirs(checkpoint_path)
            return checkpoint_path, None
        else:
            checkpoints = [f for f in os.listdir(checkpoint_path) if f.startswith('checkpoint')]
            if len(checkpoints) == 0:
                print(' ')
                output('Directory {} exists, but has not checkpoints yet'.format(checkpoint_path))
                return checkpoint_path, None
            elif len(checkpoints) == 1:
                print(' ')
                output('Directory {} exists and has only one checkpoint'.format(checkpoint_path))
                return checkpoint_path, None
            else:
                last_good_checkpoint = sorted(checkpoints, key=lambda s: int(s.split('_')[-1]))[-2]
                print(' ')
                output('Directory {} has {} checkpoints, last good checkpoint: {}'.format(
                    checkpoint_path, len(checkpoints), last_good_checkpoint))
                return checkpoint_path, path([checkpoint_path, last_good_checkpoint])


def get_epochs(args):
    if args.epochs is None:
        lst = os.listdir(args.checkpoints)
    else:
        lst = ['checkpoint_epoch_{}'.format(e) for e in args.epochs.split(',')]
    return sorted(lst, key=lambda s: int(s.split('_')[-1]))


def predictions_are_computed(checkpoint, prediction_output):
    if checkpoint not in os.listdir(prediction_output):
        return False
    if len(os.listdir(path([prediction_output, checkpoint]))) == 0:
        return False
    if len(os.listdir(path([prediction_output, checkpoint]))) == 1:
        target = os.listdir(path([prediction_output, checkpoint]))[0]
        if not os.path.isdir(path([prediction_output, checkpoint, target])):
            return False
        if len(os.listdir(path([prediction_output, checkpoint, target]))) == 0:
            return False
    return True


def get_atom_type_to_id(atom_types_path):
    with open(atom_types_path, 'r') as f:
        atom_types = f.readlines()
    return {atom_type.strip(): i for i, atom_type in enumerate(atom_types)}


def get_residue_type_to_id(atom_types_path):
    with open(atom_types_path, 'r') as f:
        atom_types = f.readlines()
    residues = sorted(list(set([atom_type.strip().split('_')[0] for atom_type in atom_types])))
    return {residue: i for i, residue in enumerate(residues)}


def get_bad_targets(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return set(map(lambda l: 'T0' + l.strip(), lines))


def check_model(model_name, model_path, include_near_native, correct_model_files_set):
    if model_name == '.DS_Store':
        return False
    if model_name in names.SUPPORTING_OBJECTS:
        return False
    if not include_near_native and names.NEAR_NATIVE_NAME_PATTERN in model_name:
        return False
    else:
        return all([f in os.listdir(model_path) for f in correct_model_files_set])


def check_target(target_name, bad_targets):
    return target_name not in bad_targets and target_name != '.DS_Store'


def iterate_models(target_path, include_near_native, correct_model_files_set):
    for obj in os.listdir(target_path):
        obj_path = path([target_path, obj])
        if check_model(obj, obj_path, include_near_native, correct_model_files_set):
            yield obj, obj_path


def iterate_targets(data_path, bad_targets=None):
    if bad_targets is None:
        bad_targets = []
    for obj in os.listdir(data_path):
        obj_path = path([data_path, obj])
        if check_target(obj, bad_targets):
            yield obj, obj_path


def calculate_z_scores(vector):
    threshold = -2
    raw_z_scores = (vector - np.mean(vector)) / np.std(vector)
    filtered_vector = [vector[i] for i, z in enumerate(raw_z_scores) if z > threshold]
    z_scores = (vector - np.mean(filtered_vector)) / np.std(filtered_vector)
    z_scores = np.clip(z_scores, -2, None)
    return z_scores


def fisher_mean(correlations):
    return np.tanh(np.arctanh(correlations).mean())


def now():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S:%f")


def output(message):
    if '\n' not in message:
        print('{} | {}'.format(now(), message))
    else:
        for line in message.split('\n'):
            print('{} | {}'.format(now(), line))


class Logger:
    def __init__(self, log_path, header, include_model=True, verbose=False):
        self.log = log_path
        self.verbose = verbose
        self.include_model = include_model
        self.start(header)

    def info(self, message, model='GLOBAL'):
        if self.include_model:
            msg = '{} | INFO\t| {}\t| {}'.format(now(), model, message)
        else:
            msg = '{} | INFO\t| {}'.format(now(), message)
        with open(self.log, 'a') as f:
            f.write(msg + '\n')
        if self.verbose:
            print(msg)

    def warn(self, message, model='GLOBAL'):
        if self.include_model:
            msg = '{} | WARN\t| {}\t| {}'.format(now(), model, message)
        else:
            msg = '{} | WARN\t| {}'.format(now(), message)
        with open(self.log, 'a') as f:
            f.write(msg + '\n')
        if self.verbose:
            print(msg)

    def error(self, message, model='GLOBAL'):
        if self.include_model:
            msg = '{} | ERROR\t| {}\t| {}'.format(now(), model, message)
        else:
            msg = '{} | ERROR\t| {}'.format(now(), message)
        with open(self.log, 'a') as f:
            f.write(msg + '\n')
        if self.verbose:
            print(msg)

    def failure(self, message, model='GLOBAL'):
        if self.include_model:
            msg = '{} | FAILURE\t| {}\t| {}'.format(now(), model, message)
        else:
            msg = '{} | FAILURE\t| {}'.format(now(), message)
        with open(self.log, 'a') as f:
            f.write(msg + '\n')
        if self.verbose:
            print(msg)

    def start(self, message, model='GLOBAL'):
        if self.include_model:
            msg = '{} | START\t| {}\t| {}'.format(now(), model, message)
        else:
            msg = '{} | START\t| {}'.format(now(), message)
        with open(self.log, 'a') as f:
            f.write(msg + '\n')
        if self.verbose:
            print(msg)

    def result(self, message, model='GLOBAL'):
        if self.include_model:
            msg = '{} | RESULT\t| {}\t| {}'.format(now(), model, message)
        else:
            msg = '{} | RESULT\t| {}'.format(now(), message)
        with open(self.log, 'a') as f:
            f.write(msg + '\n')
        if self.verbose:
            print(msg)

    def finish(self, message, model='GLOBAL'):
        if self.include_model:
            msg = '{} | FINISH\t| {}\t| {}'.format(now(), model, message)
        else:
            msg = '{} | FINISH\t| {}'.format(now(), message)
        with open(self.log, 'a') as f:
            f.write(msg + '\n')
        if self.verbose:
            print(msg)


class ProcessingException(Exception):
    pass
