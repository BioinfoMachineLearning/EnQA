# Part of the convolution layers and l-DDT computation are modified from DeepAccNet
# Improved protein structure refinement guided by deep learning based accuracy estimation
# Naozumi Hiranuma, Hahnbeom Park, Minkyung Baek,  View ORCID ProfileIvan Anishchanka, Justas Dauparas, David Baker
# Nature Communications doi: 10.1038/s41467-021-21511-x
# https://github.com/hiranumn/DeepAccNet

import torch
from torch.nn import functional as F

from network.EGNN import EGNN, EGNN_ne
from network.resnet import ResNet


def task_loss(pred, target, use_mean=True):
    l1_loss = torch.sum(torch.abs(pred - target))
    l2_loss = torch.sum((pred - target) ** 2)
    if use_mean:
        l1_loss /= pred.shape[0]
        l2_loss /= pred.shape[0]
    return l1_loss, l2_loss


def task_corr(pred, target):
    vx = pred - torch.mean(pred)
    vy = target - torch.mean(target)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return corr



class resEGNN(torch.nn.Module):

    def __init__(self,
                 dim1d=23,
                 dim2d=41,
                 bin_num=9,
                 num_chunks=5,
                 num_channel=32,
                 cmap_idx=25,
                 num_att_layers=6,
                 name=None,
                 verbose=False):
        self.dim1d = dim1d
        self.dim2d = dim2d
        self.num_chunks = num_chunks
        self.num_channel = num_channel
        self.name = name
        self.verbose = verbose
        self.bin_num = bin_num
        self.cmap_idx = cmap_idx
        self.num_att_layers = num_att_layers
        super(resEGNN, self).__init__()

        self.add_module("conv1d_1", torch.nn.Conv1d(dim1d, self.num_channel // 2, 1, padding=0, bias=True))
        self.add_module("conv2d_1",
                        torch.nn.Conv2d(self.num_channel + self.dim2d, self.num_channel, 1, padding=0, bias=True))
        self.add_module("inorm_1", torch.nn.InstanceNorm2d(self.num_channel, eps=1e-06, affine=True))
        self.add_module("base_resnet", ResNet(num_channel,
                                              self.num_chunks,
                                              "base_resnet",
                                              inorm=True,
                                              initial_projection=True,
                                              extra_blocks=False))

        self.add_module("bin_resnet", ResNet(num_channel,
                                             1,
                                             "bin_resnet",
                                             inorm=False,
                                             initial_projection=True,
                                             extra_blocks=True))
        self.add_module("conv2d_bin", torch.nn.Conv2d(self.num_channel, self.bin_num, 1, padding=0, bias=True))
        self.add_module("EGNN_layers", EGNN(in_node_nf=self.num_channel // 2 + 1,
                                            hidden_nf=self.num_channel, out_node_nf=1,
                                            in_edge_nf=self.num_channel + 9, n_layers=4))

    def calculate_LDDT(self, estogram, mask, center=4):
        # Get on the same device as indices
        device = estogram.device

        # Remove diagonal from calculation
        nres = mask.shape[-1]
        mask = torch.mul(mask, torch.ones((nres, nres), device=device) - torch.eye(nres, device=device))
        masked = torch.mul(estogram.squeeze(), mask)

        p0 = (masked[center]).sum(axis=0)
        p1 = (masked[center - 1] + masked[center + 1]).sum(axis=0)
        p2 = (masked[center - 2] + masked[center + 2]).sum(axis=0)
        p3 = (masked[center - 3] + masked[center + 3]).sum(axis=0)
        p4 = mask.sum(axis=0)

        return 0.25 * (4.0 * p0 + 3.0 * p1 + 2.0 * p2 + p3) / p4

    def forward(self, f1d, f2d, pos, el):
        len_x = f1d.shape[-1]
        out_conv1d_1 = F.elu(self._modules["conv1d_1"](f1d))
        f1d_tile = torch.cat((torch.repeat_interleave(out_conv1d_1.unsqueeze(2), repeats=len_x, dim=2),
                              torch.repeat_interleave(out_conv1d_1.unsqueeze(3), repeats=len_x, dim=3)),
                             dim=1)
        out_cat_2 = torch.cat((f1d_tile, f2d), dim=1)
        out_conv2d_1 = self._modules["conv2d_1"](out_cat_2)
        out_inorm_1 = F.elu(self._modules["inorm_1"](out_conv2d_1))

        # First ResNet
        out_base_resnet = F.elu(self._modules["base_resnet"](out_inorm_1))

        out_bin_predictor = F.elu(self._modules["bin_resnet"](out_base_resnet))
        bin_prediction = self._modules["conv2d_bin"](out_bin_predictor)
        bin_prediction = (bin_prediction + bin_prediction.permute(0, 1, 3, 2)) / 2
        estogram_prediction = F.softmax(bin_prediction, dim=1)
        lddt_prediction = self.calculate_LDDT(estogram_prediction.squeeze(), f2d[:, self.cmap_idx, :, :].squeeze())
        edge_features = torch.cat([out_base_resnet, bin_prediction], dim=1)
        edge_features = edge_features[0, :, el[0], el[1]].transpose(0, 1)
        node_features = torch.cat((out_conv1d_1.permute(0, 2, 1).squeeze(), lddt_prediction.unsqueeze(1)), dim=1)
        node_features, pos_new = self._modules["EGNN_layers"](node_features, pos, el, edge_features)
        lddt_prediction_final = lddt_prediction + torch.tanh(node_features.squeeze())
        return bin_prediction, pos_new, lddt_prediction_final


class resEGNN_with_mask(torch.nn.Module):

    def __init__(self,
                 dim1d=23,
                 dim2d=41,
                 bin_num=9,
                 num_chunks=5,
                 num_channel=32,
                 num_att_layers=6,
                 name=None,
                 verbose=False):
        self.dim1d = dim1d
        self.dim2d = dim2d
        self.num_chunks = num_chunks
        self.num_channel = num_channel
        self.name = name
        self.verbose = verbose
        self.bin_num = bin_num
        self.num_att_layers = num_att_layers
        super(resEGNN_with_mask, self).__init__()

        self.add_module("conv1d_1", torch.nn.Conv1d(dim1d, self.num_channel // 2, 1, padding=0, bias=True))
        self.add_module("conv2d_1",
                        torch.nn.Conv2d(self.num_channel + self.dim2d, self.num_channel, 1, padding=0, bias=True))
        self.add_module("inorm_1", torch.nn.InstanceNorm2d(self.num_channel, eps=1e-06, affine=True))
        self.add_module("base_resnet", ResNet(num_channel,
                                              self.num_chunks,
                                              "base_resnet",
                                              inorm=True,
                                              initial_projection=True,
                                              extra_blocks=False))

        self.add_module("bin_resnet", ResNet(num_channel,
                                             1,
                                             "bin_resnet",
                                             inorm=False,
                                             initial_projection=True,
                                             extra_blocks=True))
        self.add_module("conv2d_bin", torch.nn.Conv2d(self.num_channel, self.bin_num, 1, padding=0, bias=True))
        self.add_module("EGNN_layers", EGNN(in_node_nf=self.num_channel // 2 + 1,
                                            hidden_nf=self.num_channel, out_node_nf=1,
                                            in_edge_nf=self.num_channel + 9, n_layers=4))

    def calculate_LDDT(self, estogram, mask, center=4):
        # Get on the same device as indices
        device = estogram.device

        # Remove diagonal from calculation
        nres = mask.shape[-1]
        mask = torch.mul(mask, torch.ones((nres, nres), device=device) - torch.eye(nres, device=device))
        masked = torch.mul(estogram.squeeze(), mask)

        p0 = (masked[center]).sum(axis=0)
        p1 = (masked[center - 1] + masked[center + 1]).sum(axis=0)
        p2 = (masked[center - 2] + masked[center + 2]).sum(axis=0)
        p3 = (masked[center - 3] + masked[center + 3]).sum(axis=0)
        p4 = mask.sum(axis=0)

        return 0.25 * (4.0 * p0 + 3.0 * p1 + 2.0 * p2 + p3) / p4

    def forward(self, f1d, f2d, pos, el, cmap):
        len_x = f1d.shape[-1]
        out_conv1d_1 = F.elu(self._modules["conv1d_1"](f1d))
        f1d_tile = torch.cat((torch.repeat_interleave(out_conv1d_1.unsqueeze(2), repeats=len_x, dim=2),
                              torch.repeat_interleave(out_conv1d_1.unsqueeze(3), repeats=len_x, dim=3)),
                             dim=1)
        out_cat_2 = torch.cat((f1d_tile, f2d), dim=1)
        out_conv2d_1 = self._modules["conv2d_1"](out_cat_2)
        out_inorm_1 = F.elu(self._modules["inorm_1"](out_conv2d_1))

        # First ResNet
        out_base_resnet = F.elu(self._modules["base_resnet"](out_inorm_1))

        out_bin_predictor = F.elu(self._modules["bin_resnet"](out_base_resnet))
        bin_prediction = self._modules["conv2d_bin"](out_bin_predictor)
        bin_prediction = (bin_prediction + bin_prediction.permute(0, 1, 3, 2)) / 2
        estogram_prediction = F.softmax(bin_prediction, dim=1)
        lddt_prediction = self.calculate_LDDT(estogram_prediction.squeeze(), cmap.squeeze())
        edge_features = torch.cat([out_base_resnet, bin_prediction], dim=1)
        edge_features = edge_features[0, :, el[0], el[1]].transpose(0, 1)
        node_features = torch.cat((out_conv1d_1.permute(0, 2, 1).squeeze(), lddt_prediction.unsqueeze(1)), dim=1)
        node_features, pos_new = self._modules["EGNN_layers"](node_features, pos, el, edge_features)
        lddt_prediction_final = lddt_prediction + torch.tanh(node_features.squeeze())
        return bin_prediction, pos_new, lddt_prediction_final


class resEGNN_with_ne(torch.nn.Module):

    def __init__(self,
                 dim1d=23,
                 dim2d=41,
                 bin_num=9,
                 num_chunks=5,
                 num_channel=32,
                 cmap_idx=25,
                 num_att_layers=6,
                 name=None,
                 verbose=False):
        self.dim1d = dim1d
        self.dim2d = dim2d
        self.num_chunks = num_chunks
        self.num_channel = num_channel
        self.name = name
        self.verbose = verbose
        self.bin_num = bin_num
        self.cmap_idx = cmap_idx
        self.num_att_layers = num_att_layers
        super(resEGNN_with_ne, self).__init__()

        self.add_module("conv1d_1", torch.nn.Conv1d(dim1d, self.num_channel // 2, 1, padding=0, bias=True))
        self.add_module("conv2d_1",
                        torch.nn.Conv2d(self.num_channel + self.dim2d, self.num_channel, 1, padding=0, bias=True))
        self.add_module("inorm_1", torch.nn.InstanceNorm2d(self.num_channel, eps=1e-06, affine=True))
        self.add_module("base_resnet", ResNet(num_channel,
                                              self.num_chunks,
                                              "base_resnet",
                                              inorm=True,
                                              initial_projection=True,
                                              extra_blocks=False))

        self.add_module("bin_resnet", ResNet(num_channel,
                                             1,
                                             "bin_resnet",
                                             inorm=False,
                                             initial_projection=True,
                                             extra_blocks=True))
        self.add_module("conv2d_bin", torch.nn.Conv2d(self.num_channel, self.bin_num, 1, padding=0, bias=True))
        self.add_module("EGNN_layers", EGNN_ne(in_node_nf=self.num_channel // 2 + 1,
                                               hidden_nf=self.num_channel, out_node_nf=1,
                                               in_edge_nf=self.num_channel + 9))

    def calculate_LDDT(self, estogram, mask, center=4):
        # Get on the same device as indices
        device = estogram.device

        # Remove diagonal from calculation
        nres = mask.shape[-1]
        mask = torch.mul(mask, torch.ones((nres, nres), device=device) - torch.eye(nres, device=device))
        masked = torch.mul(estogram.squeeze(), mask)

        p0 = (masked[center]).sum(axis=0)
        p1 = (masked[center - 1] + masked[center + 1]).sum(axis=0)
        p2 = (masked[center - 2] + masked[center + 2]).sum(axis=0)
        p3 = (masked[center - 3] + masked[center + 3]).sum(axis=0)
        p4 = mask.sum(axis=0)

        return 0.25 * (4.0 * p0 + 3.0 * p1 + 2.0 * p2 + p3) / p4

    def forward(self, f1d, f2d, pos, el, cmap):
        len_x = f1d.shape[-1]
        out_conv1d_1 = F.elu(self._modules["conv1d_1"](f1d))
        f1d_tile = torch.cat((torch.repeat_interleave(out_conv1d_1.unsqueeze(2), repeats=len_x, dim=2),
                              torch.repeat_interleave(out_conv1d_1.unsqueeze(3), repeats=len_x, dim=3)),
                             dim=1)
        out_cat_2 = torch.cat((f1d_tile, f2d), dim=1)
        out_conv2d_1 = self._modules["conv2d_1"](out_cat_2)
        out_inorm_1 = F.elu(self._modules["inorm_1"](out_conv2d_1))

        # First ResNet
        out_base_resnet = F.elu(self._modules["base_resnet"](out_inorm_1))

        out_bin_predictor = F.elu(self._modules["bin_resnet"](out_base_resnet))
        bin_prediction = self._modules["conv2d_bin"](out_bin_predictor)
        bin_prediction = (bin_prediction + bin_prediction.permute(0, 1, 3, 2)) / 2
        estogram_prediction = F.softmax(bin_prediction, dim=1)
        lddt_prediction = self.calculate_LDDT(estogram_prediction.squeeze(), cmap.squeeze())
        edge_features = torch.cat([out_base_resnet, bin_prediction], dim=1)
        edge_features = edge_features[0, :, el[0], el[1]].transpose(0, 1)
        node_features = torch.cat((out_conv1d_1.permute(0, 2, 1).squeeze(), lddt_prediction.unsqueeze(1)), dim=1)
        node_features, pos_new = self._modules["EGNN_layers"](node_features, pos, el, edge_features)
        lddt_prediction_final = lddt_prediction + torch.tanh(node_features.squeeze())
        return bin_prediction, pos_new, lddt_prediction_final


if __name__ == '__main__':
    L = 300
    model = resEGNN_with_ne(dim1d=33, dim2d=25 + 45)
    model = model.cuda()
    f1d = torch.rand((1, 33, L)).cuda()
    f2d = torch.rand((1, 25 + 45, L, L)).cuda()
    pos = torch.rand((L, 3)).cuda()
    el = [torch.arange(L - 1, -1, -1).cuda(), torch.arange(L).cuda()]
    cmap = torch.rand((L, L)).cuda()
    out = model(f1d, f2d, pos, el, cmap)
    for i in out:
        print(i.shape)
