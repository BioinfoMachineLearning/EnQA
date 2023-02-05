import os
import sys

import numpy as np
import torch
import esm
from Bio.PDB import PDBParser

# refdir = "/bml/ccm3x/reference/"
# outdir = "/bml/ccm3x/dataset/attn/"

refdir = sys.argv[1]
outdir = sys.argv[2]


d3to1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

parser = PDBParser(QUIET=True)
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.eval()

for f in os.listdir(refdir):
    print(f)
    record = refdir + f
    structure = parser.get_structure('struct', record)
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
        results = model(batch_tokens, repr_layers=[6], return_contacts=True)

    token_representations = results["attentions"].numpy()
    token_representations = np.reshape(token_representations, (-1, len(seq) + 2, len(seq) + 2))
    token_representations = token_representations[:, 1: len(seq) + 1, 1: len(seq) + 1]
    outfile = outdir + f.replace('.pdb', '.npy')
    np.save(outfile, token_representations)
