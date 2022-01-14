import pandas as pd

from biopandas.pdb import PandasPdb


def create_atom_label(row):
    residue = row['residue_name']
    atom = row['atom_name']
    if atom == 'OXT':
        return '{}_{}'.format(residue, 'O')
    else:
        return '{}_{}'.format(residue, atom)


def create_atom_id_raw(chain_id, residue_number, residue_name, atom_name):
    return '{}_{}_{}_{}'.format(chain_id, residue_number, residue_name, atom_name)


def create_atom_id(row, additional=False):
    if additional is False:
        return create_atom_id_raw(
            row['chain_id'],
            int(row['residue_number']),
            row['residue_name'],
            row['atom_name'])
    else:
        return create_atom_id_raw(
            row['chain_id_'],
            int(row['residue_number_']),
            row['residue_name_'],
            row['atom_name_'])


def parse_atom_id(atom_id):
    return atom_id.split('_')[0], atom_id.split('_')[1], atom_id.split('_')[2], atom_id.split('_')[3]


def parse_residue_number(atom_id):
    return int(atom_id.split('_')[1])


def get_atom_labels(atom_types_path):
    allowed_atom_labels = set([line.strip() for line in open(atom_types_path).readlines()])
    return allowed_atom_labels


def get_bonds_types(bond_types_path):
    bonds_df = pd.read_csv(bond_types_path)
    atom_1s = bonds_df['atom_id_1']
    atom_2s = bonds_df['atom_id_2']
    residues = bonds_df['comp_id']
    orders = bonds_df['value_order']
    aromats = bonds_df['pdbx_aromatic_flag']

    single_bonds = set()
    double_bonds = set()
    aromat_bonds = set()
    for i in range(len(residues)):
        if orders[i] == 'SING':
            single_bonds.add((residues[i], atom_1s[i], atom_2s[i]))
            single_bonds.add((residues[i], atom_2s[i], atom_1s[i]))
        if orders[i] == 'DOUB':
            double_bonds.add((residues[i], atom_1s[i], atom_2s[i]))
            double_bonds.add((residues[i], atom_2s[i], atom_1s[i]))
        if aromats[i] == 'Y':
            aromat_bonds.add((residues[i], atom_1s[i], atom_2s[i]))
            aromat_bonds.add((residues[i], atom_2s[i], atom_1s[i]))

    return single_bonds, double_bonds, aromat_bonds


def get_pdb_dataframe_legend(pdb_path):
    model_biopdb = PandasPdb().read_pdb(pdb_path)
    model_df = model_biopdb.df['ATOM']
    return model_df[['chain_id', 'residue_name', 'residue_number']].drop_duplicates()
