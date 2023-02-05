import os
import argparse
import re

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import cdist

from network.resEGNN import resEGNN_with_ne
from data.loader import expand_sh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict model quality and output numpy array format.')
    parser.add_argument('--core_data', type=str, required=True, help='Path to feature data.')
    parser.add_argument('--attn', type=str, required=True, help='Path to MSA representation data.')
    parser.add_argument('--features', type=str, required=False, default="one_hot,f0,f1,f2,plddt,sh,attn",
                        help='features separated by comma.')
    parser.add_argument('--model', type=str, required=False, default="egnn", help='model type.')
    parser.add_argument('--train', type=str, required=True, help='Path to train list.')
    parser.add_argument('--validation', type=str, required=True, help='Path to validation list.')

    parser.add_argument('--output', type=str, required=True,
                        help='Path to save model weights.')
    parser.add_argument('--cpu', action='store_true', default=False, help='Force to use CPU.')
    parser.add_argument('--epochs', type=int, required=False, default=60)

    parser.add_argument('--w_dist', type=float, required=False, default=1.0, help='distance loss weight')
    parser.add_argument('--w_bin', type=float, required=False, default=1.0, help='bin loss weight')
    parser.add_argument('--w_score', type=float, required=False, default=5.0, help='score loss weight')

    parser.add_argument('--lr', type=float, required=False, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.00005, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    parser.add_argument('--ckpt', type=str, required=False, default="",
                        help='resume from checkpoint')

    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() and not args.cpu else 'cpu'
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    features = args.features.split(',')

    dim1d = 0
    dim2d = 0
    if "one_hot" in features:
        dim1d += 20
    if "f0" in features:
        dim1d += 1
    if "f1" in features:
        dim1d += 1
    if "f2" in features:
        dim1d += 1
    if "plddt" in features:
        dim1d += 1
    if "sh" in features:
        dim2d += 25
    if "attn" in features:
        dim2d += 120

    if args.model == "se3":
        from network.se3_model import se3_model
        model = se3_model(dim2d=dim2d, dim1d=dim1d)
    else:
        model = resEGNN_with_ne(dim2d=dim2d, dim1d=dim1d)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.ckpt == '':
        start_epoch = 0
    else:
        model.load_state_dict(torch.load(os.path.join(args.ckpt), map_location=device))
        start_epoch = int(re.sub('.+model_weights_epoch_(\\d+)_.+.pth', '\\1', args.ckpt)) + 1

    with open(args.train, 'r') as f:
        train_list = f.readlines()
        train_list = [i.rstrip() for i in train_list]
    with open(args.validation, 'r') as f:
        val_list = f.readlines()
        val_list = [i.rstrip() for i in val_list]

    for i in range(start_epoch, args.epochs):
        train_loss_sum = 0
        total_size = 0
        model.train()
        train_list_idx = np.random.choice(len(train_list), len(train_list), False)
        for train_idx in train_list_idx:
            sample = train_list[train_idx]
            core_sample_file = os.path.join(args.core_data, sample + '.npz')
            if not os.path.isfile(core_sample_file):
                continue
            print('Train {}'.format(sample))
            core = np.load(core_sample_file)
            x = []
            x2d = []
            if "one_hot" in features:
                x.append(core["one_hot"])
            if "f0" in features:
                x.append(core['features'][[0], :])
            if "f1" in features:
                x.append(core['features'][[1], :])
            if "f2" in features:
                x.append(core['features'][[2], :])
            if "plddt" in features:
                x.append(np.expand_dims(core['plddt'], axis=0))

            f1d = torch.tensor(np.concatenate(x, 0)).to(device)
            f1d = torch.unsqueeze(f1d, 0)

            if "sh" in features:
                x2d.append(expand_sh(core['sh_adj'], f1d.shape[2]))
            if "attn" in features:
                x2d.append(np.load(os.path.join(args.attn, sample + '.npy')))

            f2d = torch.tensor(np.concatenate(x2d, 0)).to(device)
            f2d = torch.unsqueeze(f2d, 0)
            pos = torch.tensor(core['pos_data']).to(device)
            dmap = cdist(core['pos_data'], core['pos_data'])
            el = np.where(dmap <= 0.15)
            cmap = dmap <= 0.15
            cmap = torch.tensor(cmap.astype(np.float32)).to(device)
            el = [torch.tensor(i).to(device) for i in el]

            label_lddt = torch.tensor(core['lddt_label']).to(device)
            pos_transformed = torch.tensor(core['pos_label_superpose']).to(device)
            diff_dist = dmap - cdist(core['pos_label_superpose'], core['pos_label_superpose'])
            diff_bins = np.digitize(diff_dist,
                                    np.array([-np.inf, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, np.inf]) / 100) - 1
            diff_bins = torch.tensor(diff_bins.astype(np.float32), dtype=torch.long).to(device)
            diff_bins = torch.unsqueeze(diff_bins, 0)

            pred_bin, pred_pos, pred_lddt = model(f1d, f2d, pos, el, cmap)
            loss_score = F.smooth_l1_loss(pred_lddt, label_lddt)
            loss_bin = F.cross_entropy(pred_bin, diff_bins)
            loss_dist = F.mse_loss(torch.nn.functional.pdist(pred_pos),
                                   torch.nn.functional.pdist(pos_transformed))
            total_loss = args.w_dist * loss_dist + args.w_bin * loss_bin + args.w_score * loss_score
            train_loss_sum += total_loss.detach().cpu().tolist()

            total_size += 1
            if total_size % args.batch_size == 0 or total_size == len(train_list):
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        print("Epoch: {} Train loss: {:.4f}".format(i, train_loss_sum / total_size))

        val_loss_sum = 0
        total_size = 0
        model.eval()
        val_list_idx = np.random.choice(len(val_list), len(val_list), False)
        for val_idx in val_list_idx:
            sample = val_list[val_idx]
            core_sample_file = os.path.join(args.core_data, sample + '.npz')
            if not os.path.isfile(core_sample_file):
                continue

            core = np.load(core_sample_file)
            x = []
            x2d = []
            if "one_hot" in features:
                x.append(core["one_hot"])
            if "f0" in features:
                x.append(core['features'][[0], :])
            if "f1" in features:
                x.append(core['features'][[1], :])
            if "f2" in features:
                x.append(core['features'][[2], :])
            if "plddt" in features:
                x.append(np.expand_dims(core['plddt'], axis=0))

            f1d = torch.tensor(np.concatenate(x, 0)).to(device)
            f1d = torch.unsqueeze(f1d, 0)

            if "sh" in features:
                x2d.append(expand_sh(core['sh_adj'], f1d.shape[2]))
            if "attn" in features:
                x2d.append(np.load(os.path.join(args.attn, sample + '.npy')))

            f2d = torch.tensor(np.concatenate(x2d, 0)).to(device)
            f2d = torch.unsqueeze(f2d, 0)
            pos = torch.tensor(core['pos_data']).to(device)
            dmap = cdist(core['pos_data'], core['pos_data'])
            el = np.where(dmap <= 0.15)
            cmap = dmap <= 0.15
            cmap = torch.tensor(cmap.astype(np.float32)).to(device)
            el = [torch.tensor(i).to(device) for i in el]

            label_lddt = torch.tensor(core['lddt_label']).to(device)
            pos_transformed = torch.tensor(core['pos_label_superpose']).to(device)
            diff_dist = dmap - cdist(core['pos_label_superpose'], core['pos_label_superpose'])
            diff_bins = np.digitize(diff_dist,
                                    np.array([-np.inf, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, np.inf]) / 100) - 1
            diff_bins = torch.tensor(diff_bins.astype(np.float32), dtype=torch.long).to(device)
            diff_bins = torch.unsqueeze(diff_bins, 0)
            with torch.no_grad():
                pred_bin, pred_pos, pred_lddt = model(f1d, f2d, pos, el, cmap)
            loss_score = F.smooth_l1_loss(pred_lddt, label_lddt)
            loss_bin = F.cross_entropy(pred_bin, diff_bins)
            loss_dist = F.mse_loss(torch.nn.functional.pdist(pred_pos),
                                   torch.nn.functional.pdist(pos_transformed))
            val_loss = args.w_dist * loss_dist + args.w_bin * loss_bin + args.w_score * loss_score
            val_loss_sum += val_loss.detach().cpu().tolist()
            print('Val {}, val_loss {}'.format(sample, val_loss))

            total_size += 1

        print("Epoch: {} Validation loss: {:.4f}".format(i, val_loss_sum / total_size))

        torch.save(model.state_dict(), os.path.join(args.output,
                                                    'model_weights_epoch_{}_valloss_{:.4f}.pth'.format(i,
                                                                                                       val_loss_sum / total_size)))
