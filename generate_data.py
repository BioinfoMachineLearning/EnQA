import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from biopandas.pdb import PandasPdb

from feature import create_basic_features
from data.process_label import generate_coords_transform, generate_lddt_score

# modeldir = "/bml/ccm3x/models/"
# refdir = "/bml/ccm3x/reference/"
# outdir = "/bml/ccm3x/dataset/core/"

# train_list_file = "/bml/ccm3x/dataset/revision_train.txt"

modeldir = sys.argv[1]
refdir = sys.argv[2]
outdir = sys.argv[3]

train_list_file = sys.argv[4]


with open(train_list_file, 'r') as f:
    train_list = f.readlines()


all_targets = [i.rstrip() for i in train_list]
all_targets.sort()

for target in all_targets:
    save_file = outdir+target+'.npz'
    if os.path.isfile(save_file):
        continue
    print(target)
    input_pdb = modeldir + target + '.pdb'
    native_pdb = refdir + target + '.pdb'
    one_hot, features, pos_data, sh_adj, el = create_basic_features(input_pdb, outdir)

    # label xyz
    pos_label_superpose = generate_coords_transform(native_pdb, input_pdb, outdir, transform_method="rbmt", tmalign_path=None)
    pos_label_superpose = pos_label_superpose.astype(np.float32) / 100

    # label lddt
    lddt_label = generate_lddt_score(input_pdb, native_pdb, 'utils/lddt').astype(np.float32)

    # plddt
    ppdb = PandasPdb()
    ppdb.read_pdb(input_pdb)
    plddt = ppdb.df['ATOM'][ppdb.df['ATOM']['atom_name'] == 'CA']['b_factor']
    plddt = plddt.to_numpy().astype(np.float32) / 100

    np.savez(save_file, one_hot=one_hot, features=features, pos_data=pos_data, sh_adj=sh_adj, plddt=plddt,
             pos_label_superpose=pos_label_superpose, lddt_label=lddt_label)


