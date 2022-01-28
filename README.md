# EnQA

A 3D-equivariant neural network for protein structure accuracy estimation


```
usage: python3 EnQA.py [-h] --input INPUT --output OUTPUT --method METHOD [--cpu] [--alphafold_prediction ALPHAFOLD_PREDICTION] [--alphafold_feature_cache ALPHAFOLD_FEATURE_CACHE] [--af2_pdb AF2_PDB]

Predict model quality and output NumPy array format.

optional arguments:
  -h, --help                  show this help message and exit
  --input INPUT               Path to input pdb file.
  --output OUTPUT             Path to output folder.
  --method METHOD             Prediction method, can be "EGNN_Full", "se3_Full", "EGNN_esto9" or "EGNN_covariance". Ensemble can be done listing multiple models separated by comma.
  --cpu                       Force to use CPU.
  --alphafold_prediction      Path to alphafold prediction results.               
  --alphafold_feature_cache   Optional. Can cache AlphaFold features for models of the same sequence.
  --af2_pdb AF2_PDB           Optional. PDBs from AlphaFold2 predcition for index correction with input pdb
```

## Requirements:
biopandas==0.2.9

biopython==1.79

numpy==1.21.3

pandas==1.3.4

scipy==1.7.1

torch==1.10.0

[equivariant_attention](https://github.com/FabianFuchsML/se3-transformer-public) (Optional, used by models based on SE(3)-Transformer only)

You may also need to set execution permission for utils/lddt and files under utils/SGCN/bin. 

## Example usages

Running an E(n)-equivariant model under example folder:

```
python3 EnQA.py --input example/model/6KYTP/test_model.pdb --output outputs/ --method EGNN_Full --alphafold_prediction example/alphafold_prediction/6KYTP/
```

If you want to run models based on the [SE(3)-Transformer](https://arxiv.org/abs/2006.10503), then the Python package `equivariant_attention` is required and should be installed following [Fabian's implementation](https://github.com/FabianFuchsML/se3-transformer-public).

Example:

```
python3 EnQA.py --input example/model/6KYTP/test_model.pdb --output outputs/ --method se3_Full --alphafold_prediction example/alphafold_prediction/6KYTP/  
```

## Feature generation using featurizers from Spherical graph convolutional networks 

The featurizers from Spherical graph convolutional networks (S-GCN) are used to process 3D models of proteins represented as molecular graphs.
Here we provide the voronota and spherical harmonics featurizer for Linux.

If you need to rebuild the voronota for a different system, please check out the [S-GCN Repo](https://gitlab.inria.fr/GruLab/s-gcn/-/tree/master/#voronota).

Also, there are [binaries](https://gitlab.inria.fr/GruLab/s-gcn/-/tree/master/#spherical-harmonics-featurizer) built for featurizer under a different system. (Currently, only MacOS and Linux are supported)


## Generating AlphaFold2 models for assisted quality assessment

For generating models using AlphaFold2, an installation of AlphaFold2 following its [Official Repo](https://github.com/deepmind/alphafold) is required. For our experiments, we use its original model used at CASP14 with no ensembling (--model_preset=monomer), with all genetic databases used at CASP14 (--db_preset=full_dbs), and restricts templates only to structures that were available at the start of CASP14 (--max_template_date=2020-05-14).

## Quick configuration

```
pip install biopandas biopython scipy torch==1.10.1
pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html
pip install packaging wandb line_profiler
pip install git+https://github.com/AMLab-Amsterdam/lie_learn
git clone https://github.com/FabianFuchsML/se3-transformer-public.git
cd se3-transformer-public
pip install -e .
```
