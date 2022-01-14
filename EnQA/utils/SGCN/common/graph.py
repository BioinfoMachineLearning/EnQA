import numpy as np
import os
import pandas as pd
import shutil
import subprocess
import time

from biopandas.pdb import PandasPdb
from functools import cmp_to_key

from utils.SGCN.common import checks, covalent, format, names, utils

TARGET_SCORES_DF_NAMES = ['chain_id', 'residue_number', '#2', '#3', '#4', 'residue_name', 'atom_name', 'score']
CONTACTS_DF_NAMES = [
    'chain_id',
    'residue_number',
    '#3',
    '#4',
    '#5',
    'residue_name',
    'atom_name',
    'chain_id_',
    'residue_number_',
    '#10',
    '#11',
    '#12',
    'residue_name_',
    'atom_name_',
    'area',
    '#16',
    'tags',
    '#18'
]
VOLUMES_DF_NAMES = ['chain_id', 'residue_number', '#2', '#3', '#4', 'residue_name', 'atom_name', 'volume']
SHELLING_DF_NAMES = ['chain_id', 'residue_number', '#2', '#3', '#4', 'residue_name', 'atom_name', 'shelling']


def compute_target_scores(
        cadscore_exec,
        voronota_exec,
        model_path,
        target_path,
        tmp_target_scores_path,
        tmp_target_scores_path_expanded,
        window=2,
        neighbors=1):

    target_scores_cmd = [
        cadscore_exec, '--filter-model-by-target',
        '--input-model', model_path,
        '--input-target', target_path,
        '--smoothing-window', str(window),
        '--neighborhood-depth', str(neighbors),
        '--output-residue-scores', tmp_target_scores_path
    ]
    _ = subprocess.check_output(target_scores_cmd)
    expand_cmd = 'cat {} | {} expand-descriptors > {}'.format(
        tmp_target_scores_path,
        voronota_exec,
        tmp_target_scores_path_expanded)
    subprocess.call(expand_cmd, shell=True)

    scores_df = pd.read_csv(tmp_target_scores_path_expanded, sep=' ', names=TARGET_SCORES_DF_NAMES, na_values='.')
    return scores_df[['chain_id', 'residue_number', 'residue_name', 'score']]


def run_voronota(
        voronota_exec,
        voronota_radii_file,
        model_path,
        model_df,
        atom_id_to_pos,
        tmp_balls_path,
        tmp_contacts_path,
        tmp_volumes_path,
        tmp_shelling_path,
        tmp_contacts_path_expanded,
        tmp_volumes_path_expanded,
        tmp_shelling_path_expanded):

    balls_cmd = 'cat {} | {} get-balls-from-atoms-file --radii-file {} --annotated > {}'.format(
        model_path,
        voronota_exec,
        voronota_radii_file,
        tmp_balls_path)
    contacts_volumes_cmd = \
        'cat {} | {} calculate-contacts --annotated --volumes-output {} --tag-centrality --tag-peripherial > {}'
    contacts_volumes_cmd = contacts_volumes_cmd.format(
        tmp_balls_path,
        voronota_exec,
        tmp_volumes_path,
        tmp_contacts_path)
    shelling_cmd = 'cat {} | {} x-query-contacts-depth-values > {}'.format(
        tmp_contacts_path,
        voronota_exec,
        tmp_shelling_path)
    for cmd in [balls_cmd, contacts_volumes_cmd, shelling_cmd]:
        subprocess.call(cmd, shell=True)

    # expand all descriptors
    expand_cmd = 'cat {} | {} expand-descriptors > {}'
    expand_contacts_cmd = expand_cmd.format(tmp_contacts_path, voronota_exec, tmp_contacts_path_expanded)
    expand_volumes_cmd = expand_cmd.format(tmp_volumes_path, voronota_exec, tmp_volumes_path_expanded)
    expand_shelling_cmd = expand_cmd.format(tmp_shelling_path, voronota_exec, tmp_shelling_path_expanded)
    for cmd in [expand_contacts_cmd, expand_volumes_cmd, expand_shelling_cmd]:
        subprocess.call(cmd, shell=True)

    # get from expanded contacts descriptor the contacts matrix
    contacts_df = pd.read_csv(tmp_contacts_path_expanded, sep=' ', names=CONTACTS_DF_NAMES, na_values='.')
    contact_edges = contacts_df[contacts_df['chain_id_'] != 'solvent'][[
        'chain_id', 'residue_number', 'residue_name', 'atom_name',
        'chain_id_', 'residue_number_', 'residue_name_', 'atom_name_',
        'area', 'tags']].values

    contacts_dict = {}
    for c1, res_num_1, res_name_1, a1, c2, res_num_2, res_name_2, a2, area, tags in contact_edges:
        atom_id_1 = format.create_atom_id_raw(c1, int(res_num_1), res_name_1, a1)
        atom_id_2 = format.create_atom_id_raw(c2, int(res_num_2), res_name_2, a2)
        atom_1 = atom_id_to_pos[atom_id_1]
        atom_2 = atom_id_to_pos[atom_id_2]
        if atom_1 != atom_2:
            contacts_dict.setdefault(int(atom_1), {})[int(atom_2)] = area

    # get from expanded contacts descriptor the list of atoms' sasa
    sasa_df = contacts_df.loc[contacts_df['chain_id_'] == 'solvent'][CONTACTS_DF_NAMES[:7] + ['area']]

    # get from expanded volumes descriptor the list of atoms' volumes
    volumes_df = pd.read_csv(tmp_volumes_path_expanded, sep=' ', names=VOLUMES_DF_NAMES, na_values='.')

    # get from expanded shelling descriptor the list of atoms' volumes
    shelling_df = pd.read_csv(tmp_shelling_path_expanded, sep=' ', names=SHELLING_DF_NAMES, na_values='.')

    # join all features
    on = ['chain_id', 'residue_number', 'residue_name', 'atom_name']
    features_df = model_df[on]
    features_df = pd.merge(features_df, volumes_df, on=on, how='left').fillna(0)
    features_df = pd.merge(features_df, sasa_df, on=on, how='left').fillna(0)
    features_df = pd.merge(features_df, shelling_df, on=on, how='left').fillna(0)
    features_df = features_df[on + ['volume', 'shelling', 'area']]

    return contacts_dict, features_df


def correct_matrices(
        contacts_dict,
        covalent_bonds,
        pos_to_atom_id,
        single_bonds,
        double_bonds,
        aromat_bonds,
        logger,
        model_name):

    covalent_types = []
    new_covalent_bonds = []
    contacts_to_delete = set()
    for i, j in covalent_bonds:
        atom_1 = pos_to_atom_id[i]
        atom_2 = pos_to_atom_id[j]
        chain1_name, r1_num, r1_name, a1_name = format.parse_atom_id(atom_1)
        chain2_name, r2_num, r2_name, a2_name = format.parse_atom_id(atom_2)
        bond_descriptor = (r1_name, a1_name, a2_name)

        contact = contacts_dict.get(i, dict()).get(j, 0)
        contact_pair = (i, j)
        if contact == 0:
            contact = contacts_dict.get(j, dict()).get(i, 0)
            contact_pair = (j, i)
        if contact == 0:
            msg = 'No contact in covalent bond between atoms {} and {} (positions {} and {})'
            logger.warn(msg.format(atom_1, atom_2, i, j), model_name)
            continue

        if r1_num != r2_num:
            k = 2
        elif bond_descriptor in aromat_bonds:
            k = 3
        elif bond_descriptor in single_bonds:
            k = 1
        elif bond_descriptor in double_bonds:
            k = 2
        else:
            msg = 'No appropriate covalent type for atoms {} and {} (positions {} and {})'
            logger.warn(msg.format(atom_1, atom_2, i, j), model_name)
            k = 0

        covalent_types.append([i, j, k])
        new_covalent_bonds.append([i, j, contact])
        contacts_to_delete.add(contact_pair)

    new_contacts = []
    for i, j_to_contact in contacts_dict.items():
        for j, contact in j_to_contact.items():
            if (i, j) not in contacts_to_delete:
                new_contacts.append([i, j, contact])
    return new_contacts, new_covalent_bonds, covalent_types


def compute_sequence_separation(contacts, pos_to_atom_id):
    sequence_separation = []
    for atom_pos_1, atom_pos_2, area in contacts:
        if area > 0:
            r1_num = format.parse_residue_number(pos_to_atom_id[atom_pos_1])
            r2_num = format.parse_residue_number(pos_to_atom_id[atom_pos_2])
            sequence_separation.append([atom_pos_1, atom_pos_2, abs(r2_num - r1_num) + 1])
    return sequence_separation


def compute_nolb(nolb_exec, target_file, rmsd, samples_num, total_file_prefix, target_file_prefix, nolb_log_file):
    nolb_cmd = '{} {} -r {} -m -s {} -o {} > {}'.format(
        nolb_exec,
        target_file,
        rmsd,
        samples_num,
        total_file_prefix,
        nolb_log_file
    )
    parse_cmd = \
        """grep -n 'MODEL\|ENDMDL' {}_nlb_decoys.pdb """.format(total_file_prefix) + \
        """| cut -d: -f 1 """ + \
        """| awk '{{if(NR%2) printf "sed -n %d,",$1+1; else printf "%dp {}_nlb_decoys.pdb > {}_nlb_decoy_%03d.pdb\\n", $1-1,NR/2;}}' """.format(
            total_file_prefix, target_file_prefix) + \
        """| bash -sf"""
    for cmd in [nolb_cmd, parse_cmd]:
        subprocess.call(cmd, shell=True)


def create_near_native_conformations(near_native_config, target_name, target_file_path, tmp_path, logger):
    nolb_exec = near_native_config['nolb_path']
    total_file_prefix = utils.path([tmp_path, 'total'])
    target_file_prefix = utils.path([tmp_path, target_name])
    nolb_log_path = utils.path([tmp_path, 'nolb_log'])
    try:
        compute_nolb(
            nolb_exec,
            target_file_path,
            near_native_config['rmsd'],
            near_native_config['samples_num'],
            total_file_prefix,
            target_file_prefix,
            nolb_log_path)
    except Exception as ee:
        logger.failure('Could not compute near-native conformation: {}'.format(ee))
        raise utils.ProcessingException
    res_num = len(list(filter(lambda s: s.startswith(target_name), os.listdir(tmp_path))))
    if res_num != near_native_config['samples_num']:
        logger.warn('Expected {} samples instead of {}'.format(near_native_config['samples_num'], res_num))
    return [
        file for file in os.listdir(tmp_path)
        if file.startswith('{}_nlb_decoy'.format(target_name))
    ]


def process_target_file(target_name, target_path, models_path, dest_target_path, allowed_atom_labels, logger):
    try:
        target_biopdb = PandasPdb().read_pdb(target_path)
        target_df = target_biopdb.df['ATOM']
    except Exception as ee:
        logger.error('Failed to read pdb file: {}'.format(ee))
        raise utils.ProcessingException
    if not checks.check_target(target_name, target_biopdb, target_df, models_path, allowed_atom_labels):
        logger.error('Target check failed')
        raise utils.ProcessingException

    new_target_path = utils.path([dest_target_path, 'target.pdb'])
    new_target_df = target_df[target_df.apply(
        lambda row: not checks.is_hydrogen(row['atom_name']),
        axis=1)
    ].copy(deep=True)
    if np.any(new_target_df['chain_id'].apply(lambda x: pd.isna(x) or x == '')):
        new_target_df['chain_id'] = 'A'
    new_target_df.reset_index(drop=True)
    new_target_biopdb = target_biopdb
    new_target_biopdb.df['ATOM'] = new_target_df
    new_target_biopdb.to_pdb(new_target_path)
    return new_target_path


def normalize_model(model_biopdb, model_df, target_file_path, output_model_path, allowed_atom_labels):
    if target_file_path is not None:
        target_biopdb = PandasPdb().read_pdb(target_file_path)
        target_df = target_biopdb.df['ATOM']
        target_residues = set(zip(target_df['residue_number'].values, target_df['residue_name'].values))
    else:
        target_residues = None
    new_model_df = model_df.loc[model_df.apply(
        lambda r: checks.check_model_row(r, allowed_atom_labels, target_residues),
        axis=1
    )].copy(deep=True)
    if np.any(new_model_df['chain_id'].apply(lambda x: pd.isna(x) or x == '')):
        new_model_df['chain_id'] = 'A'
    new_model_df.reset_index(drop=True)
    new_model_biopdb = model_biopdb
    new_model_biopdb.df['ATOM'] = new_model_df
    new_model_path = utils.path([output_model_path, names.MODEL_FILE_NAME])
    new_model_biopdb.to_pdb(new_model_path)
    return new_model_df, new_model_path


def create_directories(destination_path, target_name):
    destination_path = utils.path([destination_path, target_name])
    tmp_path = utils.path([destination_path, 'tmp'])
    utils.mkdir(destination_path)
    utils.mkdir(tmp_path)
    return destination_path, tmp_path


def filter_model_file(model_path, output):
    lines = open(model_path, encoding="utf8", errors='ignore').readlines()
    with open(output, 'w') as f:
        for line in filter(lambda l: not l.startswith('METHOD'), lines):
            f.write(line)


def build_atom_level_data(
        model_name,
        input_model_path,
        output_model_path,
        allowed_atom_labels,
        voronota_exec,
        voronota_radii_path,
        elements_radii_path,
        tmp_model_path,
        tmp_target_scores_path,
        tmp_target_scores_path_expanded,
        tmp_balls_path,
        tmp_contacts_path,
        tmp_volumes_path,
        tmp_shelling_path,
        tmp_contacts_path_expanded,
        tmp_volumes_path_expanded,
        tmp_shelling_path_expanded,
        single_bonds,
        double_bonds,
        aromat_bonds,
        logger,
        target_file_path=None,
        cadscore_exec=None,
        cadscore_window=2,
        cadscore_neighbors=1):

    start = time.time()

    try:
        filter_model_file(input_model_path, tmp_model_path)
    except Exception as e:
        logger.failure('While filtering model PDB file: {}'.format(e), model_name)
        raise utils.ProcessingException
    try:
        model_biopdb = PandasPdb().read_pdb(tmp_model_path)
        model_df = model_biopdb.df['ATOM']
    except Exception as e:
        logger.failure('While reading filtered model PDB file: {}'.format(e), model_name)
        raise utils.ProcessingException
    checks.check_model(model_name, model_biopdb, model_df, allowed_atom_labels, logger)

    utils.mkdir(output_model_path)
    if len(os.listdir(output_model_path)) >= 8:
        logger.info('Graph already computed', model_name)
        return

    # normalize model: rm hydrogens, unknown atoms and excess residues
    try:
        new_model_df, new_model_path = normalize_model(
            model_biopdb=model_biopdb,
            model_df=model_df,
            target_file_path=target_file_path,
            output_model_path=output_model_path,
            allowed_atom_labels=allowed_atom_labels)
    except Exception as e:
        logger.failure('While normalizing model: {}'.format(e), model_name)
        raise utils.ProcessingException

    # get atom-pos mapping, atom_id = <res_num>_<res_name>_<atom_name>
    atom_ids = new_model_df.apply(format.create_atom_id, axis=1).values
    pos_to_atom_id = dict(enumerate(atom_ids))
    atom_id_to_pos = {i: p for p, i in pos_to_atom_id.items()}

    finish = time.time()
    logger.info('Checking and preparation took {:.2f} seconds'.format(finish - start), model_name)

    # compute local CAD-scores
    scores = None
    if target_file_path is not None:
        try:
            start = time.time()
            scores = compute_target_scores(
                cadscore_exec=cadscore_exec,
                voronota_exec=voronota_exec,
                model_path=new_model_path,
                target_path=target_file_path,
                tmp_target_scores_path=tmp_target_scores_path,
                tmp_target_scores_path_expanded=tmp_target_scores_path_expanded,
                window=cadscore_window,
                neighbors=cadscore_neighbors)
            finish = time.time()
            logger.info('Computing local CAD-scores took {:.2f} seconds'.format(finish - start), model_name)
        except Exception as e:
            logger.failure('In function "compute_target_scores": {}'.format(e), model_name)

    # compute contact edges matrix and features with voronota
    try:
        start = time.time()
        contacts_dict, features_df = run_voronota(
            voronota_exec=voronota_exec,
            voronota_radii_file=voronota_radii_path,
            model_path=new_model_path,
            model_df=new_model_df,
            atom_id_to_pos=atom_id_to_pos,
            tmp_balls_path=tmp_balls_path,
            tmp_contacts_path=tmp_contacts_path,
            tmp_volumes_path=tmp_volumes_path,
            tmp_shelling_path=tmp_shelling_path,
            tmp_contacts_path_expanded=tmp_contacts_path_expanded,
            tmp_volumes_path_expanded=tmp_volumes_path_expanded,
            tmp_shelling_path_expanded=tmp_shelling_path_expanded)
        finish = time.time()
        logger.info('Runnning of Voronota took {:.2f} seconds'.format(finish - start), model_name)
    except Exception as e:
        logger.failure('In function "run_voronota": {}'.format(e), model_name)
        shutil.rmtree(output_model_path)
        raise utils.ProcessingException

    # checks that contact bonds exist
    if len(contacts_dict) == 0:
        logger.failure('No atom-level contacts were found', model_name)
        shutil.rmtree(output_model_path)
        raise utils.ProcessingException

    # compute covalent bonds matrix
    try:
        start = time.time()
        covalent_bonds = covalent.get_covalent_bonds(
            model_path=new_model_path,
            elements_radii_path=elements_radii_path,
            atom_id_to_pos=atom_id_to_pos,
            model_name=model_name)
        finish = time.time()
        logger.info('Computing of covalent bonds took {:.2f} seconds'.format(finish - start), model_name)
    except Exception as e:
        logger.failure('In function "compute_covalent_matrix": {}'.format(e), model_name)
        shutil.rmtree(output_model_path)
        raise utils.ProcessingException

    # checks that covalent bonds exist
    if len(covalent_bonds) == 0:
        logger.warn('No covalent bonds were found', model_name)
        shutil.rmtree(output_model_path)
        raise utils.ProcessingException

    # compute covalent types and rm covalent edges from contact matrix
    try:
        start = time.time()
        contacts, covalent_bonds, covalent_types = correct_matrices(
            contacts_dict=contacts_dict,
            covalent_bonds=covalent_bonds,
            pos_to_atom_id=pos_to_atom_id,
            single_bonds=single_bonds,
            double_bonds=double_bonds,
            aromat_bonds=aromat_bonds,
            logger=logger,
            model_name=model_name,)
        finish = time.time()
        logger.info('Correction of matrices took {:.2f} seconds'.format(finish - start), model_name)
    except utils.ProcessingException:
        raise utils.ProcessingException
    except Exception as e:
        logger.failure('In function "correct_matrices": {}'.format(e), model_name)
        shutil.rmtree(output_model_path)
        raise utils.ProcessingException

    # compute sequence-separation matrix
    try:
        start = time.time()
        sequence_separation = compute_sequence_separation(
            contacts=contacts,
            pos_to_atom_id=pos_to_atom_id)
        finish = time.time()
        logger.info('Computing of sequence-separation matrix took {:.2f} seconds'.format(finish - start), model_name)
    except Exception as e:
        logger.failure('In function "compute_sequence_separation": {}'.format(e), model_name)
        shutil.rmtree(output_model_path)
        raise utils.ProcessingException

    start = time.time()

    # features
    features_df['atom_label'] = features_df.apply(format.create_atom_label, axis=1)
    features_df['position'] = features_df.apply(lambda row: atom_id_to_pos[format.create_atom_id(row)], axis=1)
    features_df['sasa'] = features_df['area']
    features_df = features_df[['atom_label', 'position', 'volume', 'shelling', 'sasa']]
    features_df = features_df.rename(columns={'shelling': 'buriedness'})
    features_df = features_df.sort_values(by='position').drop(['position'], axis=1)

    # compute atoms aggregation matrix
    aggr_df = new_model_df[['record_name', 'chain_id', 'residue_number', 'residue_name', 'atom_name']].copy()
    aggr_df['position'] = aggr_df.apply(lambda row: atom_id_to_pos[format.create_atom_id(row)], axis=1)
    aggr_df = aggr_df\
        .groupby(['chain_id', 'residue_number', 'residue_name'])\
        .agg({'record_name': 'count', 'position': 'max'})\
        .reset_index()\
        .sort_values(by='position')\
        .rename(columns={'record_name': 'atoms_count'})

    # assert
    if aggr_df['atoms_count'].sum() != len(new_model_df):
        logger.error('Different atoms number in model and in aggregation matrix', model_name)
        shutil.rmtree(output_model_path)
        raise utils.ProcessingException

    aggr = aggr_df['atoms_count'].values
    finish = time.time()
    logger.info('Output atom-level files reformatting took {:.2f} seconds'.format(finish - start), model_name)

    def compare(x, y):
        if x[0] < y[0]:
            return -1
        if y[0] < x[0]:
            return 1
        if x[1] < y[1]:
            return -1
        if y[1] < x[1]:
            return 1
        return 0
    key = cmp_to_key(compare)

    # save result
    if scores is not None:
        scores.to_csv(utils.path([output_model_path, names.Y_FILE_NAME]), index=False, sep=' ')
    features_df.to_csv(utils.path([output_model_path, names.X_FILE_NAME]), index=False, sep=' ')
    np.savetxt(
        utils.path([output_model_path, names.AGGR_MASK_NAME]),
        aggr, fmt='%d')
    np.savetxt(
        utils.path([output_model_path, names.ADJ_C_NAME]),
        sorted(contacts, key=key), fmt='%d %d %f')
    np.savetxt(
        utils.path([output_model_path, names.ADJ_B_NAME]),
        sorted(covalent_bonds, key=key), fmt='%d %d %f')
    np.savetxt(
        utils.path([output_model_path, names.SEQ_SEP_NAME]),
        sorted(sequence_separation, key=key), fmt='%d %d %d')
    np.savetxt(
        utils.path([output_model_path, names.COVALENT_TYPES_NAME]),
        sorted(covalent_types, key=key), fmt='%d %d %d')


def build_residue_level_data(model_name, model_path, logger):
    adj_b_path = utils.path([model_path, names.ADJ_B_NAME])
    adj_c_path = utils.path([model_path, names.ADJ_C_NAME])
    aggr_mask_path = utils.path([model_path, names.AGGR_MASK_NAME])
    features_path = utils.path([model_path, names.X_FILE_NAME])

    # make dict atom_id -> res_id
    aggr_mask_compressed = np.loadtxt(aggr_mask_path, dtype=int)
    atom_id_to_res_id = {}
    start = 0
    for res_id, atoms_num in enumerate(aggr_mask_compressed):
        for atom_id in range(start, start + atoms_num, 1):
            atom_id_to_res_id[atom_id] = res_id
        start += atoms_num

    # make adj
    used_atoms_pairs = set()
    adj_residues_dict = {}
    adj_b_compressed = np.loadtxt(adj_b_path)
    adj_c_compressed = np.loadtxt(adj_c_path)
    adj_compressed = np.concatenate([adj_b_compressed, adj_c_compressed], axis=0)
    for a1_id, a2_id, val in adj_compressed:
        a1_id = int(a1_id)
        a2_id = int(a2_id)
        atoms_pair = tuple(sorted([a1_id, a2_id]))
        residues_pair = tuple(sorted([atom_id_to_res_id[a1_id], atom_id_to_res_id[a2_id]]))
        if atoms_pair in used_atoms_pairs:
            logger.error('Already used atom-pair {} for computing adj_res'.format(atoms_pair), model_name)
            raise utils.ProcessingException()
        if residues_pair[0] == residues_pair[1]:
            continue
        if residues_pair not in adj_residues_dict.keys():
            adj_residues_dict[residues_pair] = 0
        adj_residues_dict[residues_pair] += val

    adj_res = []
    for res_pair, val in adj_residues_dict.items():
        r1_id = int(res_pair[0])
        r2_id = int(res_pair[1])
        adj_res.append([r1_id, r2_id, val])

    if len(adj_res) < 5:
        logger.error('Small amount of contact-edges in residue-level graph: {}'.format(len(adj_res)), model_name)
        raise utils.ProcessingException
    np.savetxt(utils.path([model_path, names.ADJ_RES_NAME]), adj_res, fmt='%d %d %f')

    # make residue-level features
    features_df = pd.read_csv(features_path, sep=' ')
    curr_pos = 0
    residue_level_d = {'residue': [], 'volume': [], 'buriedness': [], 'sasa': []}
    for residue_size in aggr_mask_compressed:
        label = features_df['atom_label'].values[curr_pos].split('_')[0]
        volumes = features_df['volume'].values[curr_pos:curr_pos + residue_size]
        buriednesses = features_df['buriedness'].values[curr_pos:curr_pos + residue_size]
        sasas = features_df['sasa'].values[curr_pos:curr_pos + residue_size]
        residue_level_d['residue'].append(label)
        residue_level_d['volume'].append(np.sum(volumes))
        residue_level_d['buriedness'].append(np.min(buriednesses))
        residue_level_d['sasa'].append(np.sum(sasas))
        curr_pos += residue_size
    pd.DataFrame(residue_level_d).to_csv(utils.path([model_path, names.X_RES_FILE_NAME]), sep=' ', index=False)


def build_spherical_harmonics(model_path, order, maps_generator, skip_errors=True):
    commands = [
        maps_generator,
        '--mode', 'sh',
        '-i', os.path.join(model_path, names.MODEL_FILE_NAME),
        '-o', os.path.join(model_path, 'out_features.txt'),
        '-p', str(order)]
    if skip_errors:
        commands.append('--skip_errors')
    result = subprocess.run(commands, stdout=subprocess.PIPE)
    sh_output = result.stdout.decode('utf-8')
    lines = sh_output.split('\n')
    if 'printing' not in lines[0]:
        return
    residues_num = int(lines[0].split()[3])
    lines = lines[1:]
    features = dict()
    i = 0
    for resid1 in range(residues_num):
        for resid2 in range(residues_num):
            if resid1 != resid2:
                if features.get(resid1) is None:
                    features[resid1] = dict()
                features[resid1][resid2] = lines[i].split()
                i += 1
    adjacency = []
    with open(os.path.join(model_path, names.ADJ_RES_NAME)) as f:
        adj_lines = f.readlines()
    for line in adj_lines:
        words = line.split()
        adjacency.append((int(words[0]), int(words[1])))

    sh = []
    for source, dest in adjacency:
        sh_features = list(map(float, features[source][dest]))
        sh.append([float(source), float(dest)] + sh_features)
        sh_features = list(map(float, features[dest][source]))
        sh.append([float(dest), float(source)] + sh_features)
    sh = np.array(sh)
    np.save(os.path.join(model_path, names.SH_NAME), sh)


def preprocess_models_for_casp(
        target_name,
        models_path,
        output_path,
        allowed_atom_labels,
        elements_radii_path,
        voronota_radii_path,
        single_bonds,
        double_bonds,
        aromat_bonds,
        voronota_exec,
        target_path=None,
        near_native_config=None,
        cadscore_exec=None,
        cadscore_window=2,
        cadscore_neighbors=1,
        sh_order=None,
        maps_generator_exec=None):

    # paths
    dest_target_path, tmp_path = create_directories(output_path, target_name)
    tmp_model_path = utils.path([tmp_path, 'model'])
    tmp_target_scores_path = utils.path([tmp_path, 'scores'])
    tmp_target_scores_path_expanded = utils.path([tmp_path, 'scores_expanded'])
    tmp_balls_path = utils.path([tmp_path, 'balls'])
    tmp_contacts_path = utils.path([tmp_path, 'contacts'])
    tmp_volumes_path = utils.path([tmp_path, 'volumes'])
    tmp_shelling_path = utils.path([tmp_path, 'shelling'])
    tmp_contacts_path_expanded = utils.path([tmp_path, 'contacts_expanded'])
    tmp_volumes_path_expanded = utils.path([tmp_path, 'volumes_expanded'])
    tmp_shelling_path_expanded = utils.path([tmp_path, 'shelling_expanded'])
    log_path = utils.path([dest_target_path, names.LOG])
    logger = utils.Logger(log_path, target_name, verbose=False)

    target_file_path = None
    near_native_models = []
    if target_path is not None:
        # Read and copy target PDB-file
        try:
            target_file_path = process_target_file(
                target_name=target_name,
                target_path=target_path,
                models_path=models_path,
                dest_target_path=dest_target_path,
                allowed_atom_labels=allowed_atom_labels,
                logger=logger)
        except utils.ProcessingException:
            return
        except Exception as ee:
            logger.failure('Got unexpected exception while reading target PDB-file: {}'.format(ee))
            return

        # Compute near-native conformations
        if near_native_config is not None:
            try:
                near_native_models = create_near_native_conformations(
                    near_native_config=near_native_config,
                    target_name=target_name,
                    target_file_path=target_file_path,
                    tmp_path=tmp_path,
                    logger=logger)
            except utils.ProcessingException:
                return
            except Exception as ee:
                logger.failure('Got unexpected exception while computing near-native models: {}'.format(ee))
                return

    for model_name in near_native_models + os.listdir(utils.path([models_path, target_name])):
        if model_name.startswith('{}_nlb_decoy'.format(target_name)):
            input_model_path = utils.path([tmp_path, model_name])
            model_name = model_name.split('.')[0]
        else:
            input_model_path = utils.path([models_path, target_name, model_name])
        try:
            build_atom_level_data(
                model_name=model_name,
                input_model_path=input_model_path,
                output_model_path=utils.path([dest_target_path, model_name]),
                allowed_atom_labels=allowed_atom_labels,
                voronota_exec=voronota_exec,
                voronota_radii_path=voronota_radii_path,
                elements_radii_path=elements_radii_path,
                tmp_model_path=tmp_model_path,
                tmp_target_scores_path=tmp_target_scores_path,
                tmp_target_scores_path_expanded=tmp_target_scores_path_expanded,
                tmp_balls_path=tmp_balls_path,
                tmp_contacts_path=tmp_contacts_path,
                tmp_volumes_path=tmp_volumes_path,
                tmp_shelling_path=tmp_shelling_path,
                tmp_contacts_path_expanded=tmp_contacts_path_expanded,
                tmp_volumes_path_expanded=tmp_volumes_path_expanded,
                tmp_shelling_path_expanded=tmp_shelling_path_expanded,
                single_bonds=single_bonds,
                double_bonds=double_bonds,
                aromat_bonds=aromat_bonds,
                logger=logger,
                target_file_path=target_file_path,
                cadscore_exec=cadscore_exec,
                cadscore_window=cadscore_window,
                cadscore_neighbors=cadscore_neighbors)
        except utils.ProcessingException:
            continue
        except Exception as e:
            logger.failure('Got unexpected exception while building atom-level data: {}'.format(e), model_name)
            continue

        try:
            build_residue_level_data(
                model_name=model_name,
                model_path=utils.path([dest_target_path, model_name]),
                logger=logger)
        except utils.ProcessingException:
            continue
        except Exception as e:
            logger.failure('Got unexpected exception while building residue-level data: {}'.format(e), model_name)
            continue

        if maps_generator_exec is not None and sh_order is not None:
            try:
                build_spherical_harmonics(
                    model_path=utils.path([dest_target_path, model_name]),
                    order=sh_order,
                    maps_generator=maps_generator_exec,
                    skip_errors=True)
            except Exception as e:
                logger.failure('Got unexpected exception while building spherical harmonics: {}'.format(e), model_name)
                continue

    utils.output('{}: DONE'.format(target_name))

