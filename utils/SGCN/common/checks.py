import os

from utils.SGCN.common import utils
from utils.SGCN.common.format import create_atom_label


def is_hydrogen(atom_name):
    return ''.join([sym for sym in atom_name if not sym.isdigit()])[0] == 'H'


def check_atoms(df, allowed_atom_labels):
    df_without_hydrogens = df[df.apply(lambda row: not is_hydrogen(row['atom_name']), axis=1)]
    labels = set(df_without_hydrogens.apply(create_atom_label, axis=1).values)
    if len(labels.difference(allowed_atom_labels)) > 0:
        return False
    else:
        return True


def check_target(target_name, target_biopdb, target_df, models_path, allowed_atom_labels):
    if target_name not in os.listdir(models_path):
        utils.output('{}: no models for this target'.format(target_name))
        return False
    if not target_biopdb.df['HETATM'].empty:
        utils.output('{}: HETATM exists'.format(target_name))
        return False
    if not all(target_df['chain_id'].values == ''):
        utils.output('{}: multiple chains'.format(target_name))
        return False
    if not check_atoms(target_df, allowed_atom_labels):
        utils.output('{}: bad atoms in target pdb file'.format(target_name))
        return False
    return True


def check_model(model_name, model_biopdb, model_df, allowed_atom_labels, logger):
    if '.' in model_name:
        return False
    if not model_biopdb.df['HETATM'].empty:
        logger.warn('{} | HETATM exists in model pdb file'.format(model_name))
        return False
    if len(set(model_df['chain_id'].values)) > 1:
        logger.info('{} | multiple chain ids in model pdb file'.format(model_name))
        return False
    if not check_atoms(model_df, allowed_atom_labels):
        logger.warn('{} | bad atoms in model pdb file'.format(model_name))
        return False
    return True


def check_model_row(row, allowed_atom_labels, target_residues=None):
    residue = (row['residue_number'], row['residue_name'])
    correct_atom = not is_hydrogen(row['atom_name']) and create_atom_label(row) in allowed_atom_labels
    return correct_atom and (target_residues is None or residue in target_residues)


