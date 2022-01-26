# Code modified from https://github.com/FabianFuchsML/se3-transformer-public
# @inproceedings{fuchs2020se3transformers,
#    title={SE(3)-Transformers: 3D Roto-Translation Equivariant Attention Networks},
#    author={Fabian B. Fuchs and Daniel E. Worrall and Volker Fischer and Max Welling},
#    year={2020},
#    booktitle = {Advances in Neural Information Processing Systems 34 (NeurIPS)},
#}


import dgl
import torch
import torch.nn as nn
from torch.nn import functional as F

from equivariant_attention.modules import GNormBias, get_basis_and_r, GSE3Res
from equivariant_attention.fibers import Fiber

from network.resnet import ResNet

class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""

    def __init__(self, num_layers=2, num_channels=16, num_degrees=3, n_heads=4, div=4,
                 si_m='1x1', si_e='att',
                 l0_in_features=32, l0_out_features=1,
                 l1_in_features=1, l1_out_features=1,
                 num_edge_features=32, x_ij=None):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = num_edge_features
        self.div = div
        self.n_heads = n_heads
        self.si_m, self.si_e = si_m, si_e
        self.x_ij = x_ij

        if l1_out_features > 0:
            fibers = {'in': Fiber(dictionary={0: l0_in_features, 1: l1_in_features}),
                      'mid': Fiber(self.num_degrees, self.num_channels),
                      'out': Fiber(dictionary={0: l0_out_features, 1: l1_out_features})}
        else:
            fibers = {'in': Fiber(dictionary={0: l0_in_features, 1: l1_in_features}),
                      'mid': Fiber(self.num_degrees, self.num_channels),
                      'out': Fiber(dictionary={0: l0_out_features})}

        blocks = self._build_gcn(fibers)
        self.Gblock = blocks

    def _build_gcn(self, fibers):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim,
                                  div=self.div, n_heads=self.n_heads,
                                  learnable_skip=True, skip='cat',
                                  selfint=self.si_m, x_ij=self.x_ij))
            Gblock.append(GNormBias(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(
            GSE3Res(fibers['mid'], fibers['out'], edge_dim=self.edge_dim,
                    div=1, n_heads=min(1, 2), learnable_skip=True,
                    skip='cat', selfint=self.si_e, x_ij=self.x_ij))
        return nn.ModuleList(Gblock)

    def forward(self, G, type_0_features, type_1_features):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees - 1)
        h = {'0': type_0_features, '1': type_1_features}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)
        return h['0'], h['1'].squeeze()


class se3_model(torch.nn.Module):

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
        super(se3_model, self).__init__()

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
        self.add_module("se3_layers", SE3Transformer(l0_in_features=self.num_channel // 2 + 1,
                                                     num_edge_features=self.num_channel + 9))

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
        edge_features = edge_features[0, :, el[0], el[1]].transpose(0, 1)  # (num_e,d)
        node_features = torch.cat((out_conv1d_1.permute(0, 2, 1).squeeze(), lddt_prediction.unsqueeze(1)),
                                  dim=1)  # (b,L,d)

        g = dgl.graph((el[0], el[1]), num_nodes=pos.shape[0])
        pos = pos.reshape(pos.shape[0], 1, -1)
        g.edata['d'] = pos[el[0], 0, :] - pos[el[1], 0, :]
        g.edata['w'] = edge_features
        node_features = node_features.reshape((node_features.shape[0], node_features.shape[1], 1))

        node_features, pos_new = self._modules["se3_layers"](g, node_features, pos)
        lddt_prediction_final = lddt_prediction + torch.tanh(node_features.squeeze())
        return bin_prediction, pos_new, lddt_prediction_final
