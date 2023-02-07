import os
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from network.resEGNN import resEGNN_with_ne

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict model quality and output numpy array format.')
    parser.add_argument('--train', type=str, required=True,
                        help='Path to train feature dataset.')
    parser.add_argument('--validation', type=str, required=True,
                        help='Path to validation feature dataset.')
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

    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() and not args.cpu else 'cpu'
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    dim2d = 25 + 9 * 5
    model = resEGNN_with_ne(dim2d=dim2d, dim1d=33)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    train_list = [s for s in os.listdir(args.train) if s.endswith('.pt')]

    for i in range(args.epochs):
        train_loss_sum = 0
        total_size = 0
        model.train()
        for sample in train_list:
            x = torch.load(args.train + '/' + sample)
            f1d = x['f1d'].to(device)
            f2d = x['f2d'].to(device)
            pos = x['pos'].to(device)
            el = [i.to(device) for i in x['el']]
            cmap = x['cmap'].to(device)

            label_lddt = x['label_lddt'].to(device)
            diff_bins = x['diff_bins'].to(device)
            pos_transformed = x['pos_transformed'].to(device)

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
        for sample in os.listdir(args.validation):
            if not sample.endswith('.pt'):
                continue

            x = torch.load(args.validation + '/' + sample)
            f1d = x['f1d'].to(device)
            f2d = x['f2d'].to(device)
            pos = x['pos'].to(device)
            el = [i.to(device) for i in x['el']]
            cmap = x['cmap'].to(device)

            label_lddt = x['label_lddt'].to(device)
            diff_bins = x['diff_bins'].to(device)
            pos_transformed = x['pos_transformed'].to(device)
            with torch.no_grad():
                pred_bin, pred_pos, pred_lddt = model(f1d, f2d, pos, el, cmap)

            loss_score = F.smooth_l1_loss(pred_lddt, label_lddt)
            loss_bin = F.cross_entropy(pred_bin, diff_bins)
            loss_dist = F.mse_loss(torch.nn.functional.pdist(pred_pos),
                                   torch.nn.functional.pdist(pos_transformed))
            total_loss = args.w_dist * loss_dist + args.w_bin * loss_bin + args.w_score * loss_score

            val_loss_sum += total_loss.detach().cpu()
            total_size += 1

        print("Epoch: {} Validation loss: {:.4f}".format(i, val_loss_sum / total_size))

    torch.save(model.state_dict(), os.path.join(args.output, 'model_weights.pth'))

# python3 train.py --train outputs/processed/ --validation outputs/processed/ --output outputs/ --epochs 15
