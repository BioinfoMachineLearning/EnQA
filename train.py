import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
import horovod.torch as hvd
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

from data.loader import data_generator
from network.resEGNN import resEGNN, resEGNN_with_mask
from network.se3_model import se3_model

parser = argparse.ArgumentParser(description='PyTorch ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_data_dir', type=str, required=False,
                    help='The path to model features',
                    default='example/processed')
parser.add_argument('--target_data_dir', type=str, required=False,
                    help='The path to target features.',
                    default='example/processed')
parser.add_argument('--out_path', type=str, required=False,
                    help='Output path.',
                    default='outputs/train')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--train_list_file', type=str, required=False,
                    help='Target list for training as text file.',
                    default='example/train_list.txt')
parser.add_argument('--val_list_file', type=str, required=False,
                    help='Target list for validation as text file.',
                    default='example/val_list.txt')

parser.add_argument('--save_interval', type=int, required=False,
                    help='Interval of epochs for saving models.',
                    default=1)
parser.add_argument('--disto_type', type=str, required=False,
                    default='cov25')
parser.add_argument('--model_type', type=str, required=False, default='egnn')
parser.add_argument('--lr', type=float, required=False, help='learning rate.', default=0.001)
parser.add_argument('--w_diff_loss', type=float, required=False, default=1.0)
parser.add_argument('--w_bin_loss', type=float, required=False, default=1.0)
parser.add_argument('--w_lddt_loss', type=float, required=False, default=1.0)

parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before '
                         'executing allreduce across workers; it multiplies '
                         'total batch size.')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')

# Default settings from https://arxiv.org/abs/1706.02677.
parser.add_argument('--batch-size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=1,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')
parser.add_argument('--score_loss', type=str, required=False, default='mse')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--optimizer', type=str, required=False, default='sgd')

parser.add_argument('--max_seq_len', type=int, required=False, default=1000)
parser.add_argument('--diff_cutoff', type=int, required=False,
                    help='Max distance cutoff for edges in a graph.',
                    default=15)
parser.add_argument('--coordinate_factor', type=float, required=False, default=0.01)

args = parser.parse_args()


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    lddt_loss_log = Metric('lddt_loss')
    bin_loss_log = Metric('bin_loss')
    diff_loss_log = Metric('diff_loss')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, sample_data in enumerate(train_loader):
            # adjust_learning_rate(epoch, batch_idx)
            optimizer.zero_grad()

            f1d, f2d, pos, el, cmap, diff_bins, pos_label_superpose, lddt_label = sample_data
            cmap = cmap.squeeze()
            cmap = cmap.cuda()
            f1d = f1d.cuda()
            f2d = f2d.cuda()
            y_diff = y_diff.cuda()
            y_bin = y_bin.cuda()



            pos = sample_data['position'].squeeze()
            pos = pos.cuda()

            if f1d.shape[2] > args.max_seq_len:
                start_idx = np.random.choice(range(0, f1d.shape[2] - args.max_seq_len), 1)[0]
                end_idx = start_idx + args.max_seq_len
                f1d = f1d[:, :, start_idx:end_idx]
                f2d = f2d[:, :, start_idx:end_idx, :][:, :, :, start_idx:end_idx]
                pos = pos[start_idx:end_idx, :]
                el = [i.squeeze().cuda() for i in torch.where(torch.cdist(pos, pos) <= args.diff_cutoff * args.coordinate_factor)]
                cmap = cmap[start_idx:end_idx, :][:, start_idx:end_idx]
            else:
                start_idx = 0
                end_idx = f1d.shape[2]
                el = [i.squeeze().cuda() for i in sample_data['el']]
            # f1d: (b, d, L)
            # f2d: (b, d, L, L)
            # pos: (L,3)
            # el: (E, E)
            # cmap: (L, L)
            pred_bin, pos_new, pred_lddt = model(f1d, f2d, pos, el, cmap)
            # score loss
            # print(sample_data['lddt_mask'].shape)
            # print(pred_lddt.shape)
            # print(y_lddt)
            y_lddt = y_lddt[start_idx:end_idx]
            if args.loss == 'l1loss':
                lddt_loss = F.smooth_l1_loss(pred_lddt[sample_data['lddt_mask'][0, start_idx:end_idx]],
                                             y_lddt[sample_data['lddt_mask'][0, start_idx:end_idx]])
            else:
                lddt_loss = F.mse_loss(pred_lddt[sample_data['lddt_mask'][0, start_idx:end_idx]],
                                       y_lddt[sample_data['lddt_mask'][0, start_idx:end_idx]])
            # disto bin loss

            pred_filter = pred_bin[:, :, start_idx:end_idx, :][:, :, :, start_idx:end_idx]
            y_bin = y_bin[:, start_idx:end_idx, :][:, :, start_idx:end_idx]

            pred_filter = pred_filter.reshape(pred_filter.shape[0], pred_filter.shape[1], -1)
            y_bin = y_bin.reshape(y_bin.shape[0], -1)
            loss_bin = F.cross_entropy(pred_filter, y_bin)
            # position loss
            pred_val_filter = pos_new[start_idx:end_idx, :]
            y_superpose = sample_data['superpose'].squeeze().cuda()
            y_superpose = y_superpose[start_idx:end_idx, :]
            if args.use_dist_loss:
                loss_diff = F.mse_loss(torch.nn.functional.pdist(pred_val_filter, 2),
                                       torch.nn.functional.pdist(y_superpose, 2))
            else:
                loss_diff = F.mse_loss(pred_val_filter, y_superpose)
            total_loss = args.w_diff_loss * loss_diff + args.w_bin_loss * loss_bin + args.w_lddt_loss * lddt_loss

            assert not torch.any(torch.isnan(total_loss))
            lddt_loss_log.update(lddt_loss)
            bin_loss_log.update(loss_bin)
            diff_loss_log.update(loss_diff)

            total_loss.backward()
            optimizer.step()
            t.set_postfix({'loss': lddt_loss_log.avg.item()})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss_lddt', lddt_loss_log.avg, epoch)
        log_writer.add_scalar('train/loss_bin', bin_loss_log.avg, epoch)
        log_writer.add_scalar('train/loss_diff', diff_loss_log.avg, epoch)


def validate(epoch):
    val_sampler.set_epoch(epoch)
    model.eval()
    val_lddt_loss_log = Metric('val_lddt_loss')
    val_bin_loss_log = Metric('val_bin_loss')
    val_diff_loss_log = Metric('val_diff_loss')


    with tqdm(total=len(val_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():

            for batch_idx, sample_data in enumerate(val_loader):
                # adjust_learning_rate(epoch, batch_idx)
                optimizer.zero_grad()

                f1d, f2d, pos, el, cmap, diff_bins, pos_label_superpose, lddt_label = sample_data
                cmap = cmap.squeeze()
                cmap = cmap.cuda()
                f1d = f1d.cuda()
                f2d = f2d.cuda()
                y_diff = y_diff.cuda()
                y_bin = y_bin.cuda()



                pos = sample_data['position'].squeeze()
                pos = pos.cuda()

                if f1d.shape[2] > args.max_seq_len:
                    start_idx = np.random.choice(range(0, f1d.shape[2] - args.max_seq_len), 1)[0]
                    end_idx = start_idx + args.max_seq_len
                    f1d = f1d[:, :, start_idx:end_idx]
                    f2d = f2d[:, :, start_idx:end_idx, :][:, :, :, start_idx:end_idx]
                    pos = pos[start_idx:end_idx, :]
                    el = [i.squeeze().cuda() for i in torch.where(torch.cdist(pos, pos) <= args.diff_cutoff * args.coordinate_factor)]
                    cmap = cmap[start_idx:end_idx, :][:, start_idx:end_idx]
                else:
                    start_idx = 0
                    end_idx = f1d.shape[2]
                    el = [i.squeeze().cuda() for i in sample_data['el']]
                # f1d: (b, d, L)
                # f2d: (b, d, L, L)
                # pos: (L,3)
                # el: (E, E)
                # cmap: (L, L)
                pred_bin, pos_new, pred_lddt = model(f1d, f2d, pos, el, cmap)
                # score loss
                # print(sample_data['lddt_mask'].shape)
                # print(pred_lddt.shape)
                # print(y_lddt)
                y_lddt = y_lddt[start_idx:end_idx]
                if args.loss == 'l1loss':
                    lddt_loss = F.smooth_l1_loss(pred_lddt[sample_data['lddt_mask'][0, start_idx:end_idx]],
                                                 y_lddt[sample_data['lddt_mask'][0, start_idx:end_idx]])
                else:
                    lddt_loss = F.mse_loss(pred_lddt[sample_data['lddt_mask'][0, start_idx:end_idx]],
                                           y_lddt[sample_data['lddt_mask'][0, start_idx:end_idx]])
                # disto bin loss

                pred_filter = pred_bin[:, :, start_idx:end_idx, :][:, :, :, start_idx:end_idx]
                y_bin = y_bin[:, start_idx:end_idx, :][:, :, start_idx:end_idx]

                pred_filter = pred_filter.reshape(pred_filter.shape[0], pred_filter.shape[1], -1)
                y_bin = y_bin.reshape(y_bin.shape[0], -1)
                loss_bin = F.cross_entropy(pred_filter, y_bin)
                # position loss
                pred_val_filter = pos_new[start_idx:end_idx, :]
                y_superpose = sample_data['superpose'].squeeze().cuda()
                y_superpose = y_superpose[start_idx:end_idx, :]
                if args.use_dist_loss:
                    loss_diff = F.mse_loss(torch.nn.functional.pdist(pred_val_filter, 2),
                                           torch.nn.functional.pdist(y_superpose, 2))
                else:
                    loss_diff = F.mse_loss(pred_val_filter, y_superpose)
                total_loss = args.w_diff_loss * loss_diff + args.w_bin_loss * loss_bin + args.w_lddt_loss * lddt_loss

                assert not torch.any(torch.isnan(total_loss))
                val_lddt_loss_log.update(lddt_loss)
                val_bin_loss_log.update(loss_bin)
                val_diff_loss_log.update(loss_diff)
                t.set_postfix({'loss': val_lddt_loss_log.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss_lddt', val_lddt_loss_log.avg, epoch)
        log_writer.add_scalar('val/loss_bin', val_bin_loss_log.avg, epoch)
        log_writer.add_scalar('val/loss_diff', val_diff_loss_log.avg, epoch)


def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, os.path.join(args.out_path, filepath))


if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    num_workers = 1
    allreduce_batch_size = args.batch_size * args.batches_per_allreduce

    hvd.init()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    for try_epoch in range(args.epochs, 0, -1):
        if os.path.exists(os.path.join(args.out_path, args.checkpoint_format.format(epoch=try_epoch))):
            resume_from_epoch = try_epoch
            break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()

    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0
    if not os.path.isdir(args.out_path) and hvd.rank() == 0:
        os.mkdir(args.out_path)

    # Horovod: write TensorBoard logs on first worker.
    log_writer = SummaryWriter(os.path.join(args.out_path, args.log_dir)) if hvd.rank() == 0 else None

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(num_workers)

    kwargs = {'num_workers': num_workers, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe

    kwargs['multiprocessing_context'] = 'forkserver'

    # Prepare data
    train_list = pd.read_csv(args.train_list_file, header=None)[0].to_list()
    val_list = pd.read_csv(args.val_list_file, header=None)[0].to_list()

    train_dataset = data_generator(args.model_data_dir, args.target_data_dir,
                                   disto_type=args.disto_type, target_list=train_list)
    val_dataset = data_generator(args.model_data_dir, args.target_data_dir,
                                 disto_type=args.disto_type, target_list=val_list)

    # Horovod: use DistributedSampler to partition data among workers. Manually specify
    # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=allreduce_batch_size,
        sampler=train_sampler, **kwargs)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                             sampler=val_sampler, **kwargs)
    if args.disto_type == 'disto':
        dim2d = 25 + 64 * 5
    elif args.disto_type == 'cov25':
        dim2d = 25 + 25
    else:
        dim2d = 25 + 64
    if args.model_type == 'egnn':
        model = resEGNN_with_mask(dim2d=dim2d, dim1d=33)
    elif args.model_type == 'se3':
        model = se3_model(dim2d=dim2d, dim1d=33)
    else:
        raise NotImplementedError

    lr_scaler = args.batches_per_allreduce * hvd.size() if not args.use_adasum else 1
    model.cuda()
    # If using GPU Adasum allreduce, scale learning rate by local_size.
    if args.use_adasum and hvd.nccl_built():
        lr_scaler = args.batches_per_allreduce * hvd.local_size()

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=(args.base_lr * lr_scaler),
                              momentum=args.momentum, weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=(args.base_lr * lr_scaler),
                               weight_decay=args.wd)

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=args.batches_per_allreduce,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if resume_from_epoch > 0 and hvd.rank() == 0:
        filepath = os.path.join(args.out_path, args.checkpoint_format.format(epoch=resume_from_epoch))
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    for epoch in range(resume_from_epoch, args.epochs):
        train(epoch)
        validate(epoch)
        save_checkpoint(epoch)
