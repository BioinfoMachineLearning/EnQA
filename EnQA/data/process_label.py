import os
import subprocess
from io import StringIO

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix


def parse_pdbfile(pdbfile):
    file = open(pdbfile, "r")
    lines = file.readlines()
    file.close()

    lines = [l for l in lines if l.startswith("ATOM")]
    output = {}
    for line in lines:
        if line[13] != "H":
            aidx = int(line[6:11])
            aname = line[12:16].strip()
            rname = line[17:20].strip()
            cname = line[21].strip()
            rindex = int(line[22:26])
            xcoord = float(line[30:38])
            ycoord = float(line[38:46])
            zcoord = float(line[46:54])
            occupancy = float(line[54:60])

            temp = dict(aidx=aidx,
                        aname=aname,
                        rname=rname,
                        cname=cname,
                        rindex=rindex,
                        x=xcoord,
                        y=ycoord,
                        z=zcoord,
                        occupancy=occupancy)

            residue = output.get(rindex, {})
            residue[aname] = temp
            output[rindex] = residue

    output2 = []
    keys = [i for i in output.keys()]
    keys.sort()
    for k in keys:
        temp = output[k]
        temp["rindex"] = k
        temp["rname"] = temp["CA"]["rname"]
        output2.append(temp)

    return output2


def get_coords(pose):
    nres = len(pose)

    # three anchor atoms to build local reference frame
    N = np.stack([np.array([pose[i]["N"]["x"], pose[i]["N"]["y"], pose[i]["N"]["z"]]) if "N" in pose[
        i].keys() else np.array([pose[i]["CA"]["x"], pose[i]["CA"]["y"], pose[i]["CA"]["z"]]) for i in range(nres)])
    Ca = np.stack([np.array([pose[i]["CA"]["x"], pose[i]["CA"]["y"], pose[i]["CA"]["z"]]) for i in range(nres)])
    C = np.stack([np.array([pose[i]["C"]["x"], pose[i]["C"]["y"], pose[i]["C"]["z"]]) if "C" in pose[
        i].keys() else np.array([pose[i]["CA"]["x"], pose[i]["CA"]["y"], pose[i]["CA"]["z"]]) for i in range(nres)])

    # recreate Cb given N,Ca,C
    ca = -0.58273431
    cb = 0.56802827
    cc = -0.54067466

    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = ca * a + cb * b + cc * c

    return N, Ca, C, Ca + Cb

def get_coords_ca(pose):
    nres = len(pose)
    Ca = np.stack([np.array([pose[i]["CA"]["x"], pose[i]["CA"]["y"], pose[i]["CA"]["z"]]) for i in range(nres)])
    return Ca


def init_pose(pose):
    pdict = {}
    pdict['pose'] = pose
    pdict['nres'] = len(pose)
    pdict['N'], pdict['Ca'], pdict['C'], pdict['Cb'] = get_coords(pose)


def get_distmaps(pose, atom1="CA", atom2="CA", default="CA"):
    psize = len(pose)
    xyz1 = np.zeros((psize, 3))
    xyz2 = np.zeros((psize, 3))
    for i in range(psize):
        r = pose[i]

        if type(atom1) == str:
            if atom1 in r:
                xyz1[i - 1, :] = np.array([r[atom1]["x"], r[atom1]["y"], r[atom1]["z"]])
            else:
                xyz1[i - 1, :] = np.array([r[default]["x"], r[default]["y"], r[default]["z"]])
        else:
            temp = atom1.get(r["rname"], default)
            xyz1[i - 1, :] = np.array([r[temp]["x"], r[temp]["y"], r[temp]["z"]])

        if type(atom2) == str:
            if atom2 in r:
                xyz2[i - 1, :] = np.array([r[atom2]["x"], r[atom2]["y"], r[atom2]["z"]])
            else:
                xyz2[i - 1, :] = np.array([r[default]["x"], r[default]["y"], r[default]["z"]])
        else:
            temp = atom2.get(r["rname"], default)
            xyz2[i - 1, :] = np.array([r[temp]["x"], r[temp]["y"], r[temp]["z"]])

    return distance_matrix(xyz1, xyz2)


def generate_lddt_score(input_model_path, target_path, lddt_cmd):
    proc = subprocess.Popen([lddt_cmd, input_model_path, target_path], stdout=subprocess.PIPE)
    output = proc.stdout.read().decode("utf-8")
    df = pd.read_csv(StringIO(output), sep="\t", skiprows=10)
    score = df['Score'].to_numpy()
    if score.dtype.name == 'object':
        score[score == '-'] = -1
    score = score.astype(np.float32)
    return score


def generate_dist_diff(model_name, ref_name,
                       bins=np.array([-np.inf, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, np.inf])):
    pose = parse_pdbfile(model_name)
    d_model = get_distmaps(pose, atom1="CB", atom2="CB", default="CA")
    model_idx = np.array([i['rindex'] for i in pose])

    pose_ref = parse_pdbfile(ref_name)
    d_ref = get_distmaps(pose_ref, atom1="CB", atom2="CB", default="CA")
    ref_idx = np.array([i['rindex'] for i in pose_ref])

    assert np.array_equal(ref_idx, model_idx)
    diff_dist = d_model - d_ref
    diff_bins = np.digitize(diff_dist, bins) - 1
    return diff_bins


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B
    # new superposed X should be RA+t
    return R, t


def tmalign_transform(pdb1, pdb2, temp_path, tmalign_path='utils/TMalign'):
    proc = subprocess.run('{} {} {} -m {}'.format(tmalign_path, pdb1, pdb2, temp_path), shell=True)
    f = open(temp_path, "r")
    m = np.array([i.strip().split() for i in f.readlines()[2:5]], dtype=np.float32)
    f.close()
    t, R = m[:, [1]], m[:, 2:5]
    os.remove(temp_path)
    return R, t


def generate_coords_transform(ref_pdb, model_pdb, output_feature_path, transform_method='rbmt',
                              tmalign_path='utils/TMalign'):
    pose1 = parse_pdbfile(ref_pdb)
    _, pos1, _, _ = get_coords(pose1)
    r_idx1 = np.array([i['rindex'] for i in pose1])
    pose2 = parse_pdbfile(model_pdb)
    _, pos2, _, _ = get_coords(pose2)
    r_idx2 = np.array([i['rindex'] for i in pose2])
    assert np.array_equal(r_idx2, r_idx1)
    if transform_method == 'rbmt':
        R1, t1 = rigid_transform_3D(pos1.T, pos2.T)
        pos_transformed = np.matmul(R1, pos1.T) + t1
    elif transform_method == 'tmalign':
        R2, t2 = tmalign_transform(ref_pdb, model_pdb,
                                   os.path.join(output_feature_path, 'rt_tmalign.txt'), tmalign_path)
        pos_transformed = np.matmul(R2, pos1.T) + t2
    else:
        raise ValueError
    pos_transformed = pos_transformed.transpose(1, 0).astype(np.float32)
    return pos_transformed

