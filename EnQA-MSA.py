import os

import torch
import argparse
import numpy as np
import esm
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from biopandas.pdb import PandasPdb

from data.loader import expand_sh
from feature import create_basic_features
from network.resEGNN import resEGNN_with_ne

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict model quality and output numpy array format.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input pdb file.')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output folder.')

    args = parser.parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    one_hot, features, pos_data, sh_adj, el = create_basic_features(args.input, args.output)

    # plddt
    ppdb = PandasPdb()
    ppdb.read_pdb(args.input)
    plddt = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']['b_factor']
    plddt = plddt.to_numpy().astype(np.float32) / 100

    dim1d = 24
    dim2d = 145

    parser = PDBParser(QUIET=True)
    d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
             'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
             'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
             'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


    model_esm, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model_esm.eval()
    structure = parser.get_structure('struct', args.input)
    for m in structure:
        for chain in m:
            seq = []
            for residue in chain:
                seq.append(d3to1[residue.resname])

    # Load ESM-2 model
    batch_converter = alphabet.get_batch_converter()
    data = [("protein1", ''.join(seq))]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[6], return_contacts=True)

    token_representations = results["attentions"].numpy()
    token_representations = np.reshape(token_representations, (-1, len(seq) + 2, len(seq) + 2))
    token_representations = token_representations[:, 1: len(seq) + 1, 1: len(seq) + 1]

    model = resEGNN_with_ne(dim2d=dim2d, dim1d=dim1d)

    model.to(device)
    model.load_state_dict(torch.load("models/EnQA-MSA.pth", map_location=device))
    model.eval()

    pred_lddt = []
    af2_plddt = []
    true_lddt = []

    x = [one_hot, features, np.expand_dims(plddt, axis=0)]
    f1d = torch.tensor(np.concatenate(x, 0)).to(device)
    f1d = torch.unsqueeze(f1d, 0)

    x2d = [expand_sh(sh_adj, f1d.shape[2]), token_representations]
    f2d = torch.tensor(np.concatenate(x2d, 0)).to(device)
    f2d = torch.unsqueeze(f2d, 0)
    pos = torch.tensor(pos_data).to(device)
    dmap = cdist(pos_data, pos_data)
    el = np.where(dmap <= 0.15)
    cmap = dmap <= 0.15
    cmap = torch.tensor(cmap.astype(np.float32)).to(device)
    el = [torch.tensor(i).to(device) for i in el]
    with torch.no_grad():
        _, _, lddt_pred = model(f1d, f2d, pos, el, cmap)

    out = lddt_pred.cpu().detach().numpy().astype(np.float16)
    np.save(os.path.join(args.output, os.path.basename(args.input).replace('.pdb', '.npy')), out)
