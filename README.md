# EnQA

A 3D-equivariant neural network for protein structure accuracy estimation.

## Requirements:
```
biopandas==0.3.0dev0
biopython==1.79
numpy==1.21.3
pandas==1.3.4
scipy==1.7.1
torch==1.10.0
```

[equivariant_attention](https://github.com/FabianFuchsML/se3-transformer-public) (Optional, used by models based on SE(3)-Transformer only)

[pdb-tools](https://github.com/haddocking/pdb-tools) (Optional, used by models with multiple chains only)

You may also need to set execution permission for utils/lddt and files under utils/SGCN/bin. 

Note: Currently, the dependencies support AMD/Intel based system with Ubuntu 21.10 (Impish Indri). Other Linux-based system may be also supported but not guaranteed.

Install [Transformer protein language models](https://github.com/facebookresearch/esm) by the following command:

```
pip install git+https://github.com/facebookresearch/esm.git
```

## EnQA-MSA

```
usage: EnQA-MSA.py [-h] --input INPUT --output OUTPUT

Predict model quality and output numpy array format.

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Path to input pdb file.
  --output OUTPUT  Path to output folder.
```

Provide input PDB from AlphaFold2 prediction with plddt stored in the "B-factor" column, then run EnQA-MSA with the following example:

```
python EnQA-MSA.py --input example/enqa-msa/1A09A.pdb --output example/output/
```

### Model training for EnQA-MSA
First, generate feature files and embeddings from MSA-Transformer. Example for [data_list_file](https://github.com/BioinfoMachineLearning/EnQA/blob/main/data/af2_train.txt) which contains list for models.

```
python3 generate_data.py <input pdb folder> <reference pdb folder> <feature save folder> <data_list_file> 
python3 generate_embedding.py <reference pdb folder> <embedding save folder>
```

After all feature files are generated,  here is how to train the model:
```
python3 train_enqa_msa.py --core_data <feature save folder> --attn <embedding save folder> --train <data_list_file for training> --validation <data_list_file for validation> --output <model_save_folder> --epochs 60
```


## EnQA assisted with AlphaFold2

```
usage: python3 EnQA.py [-h] --input INPUT --output OUTPUT --method METHOD [--cpu] [--alphafold_prediction ALPHAFOLD_PREDICTION] [--alphafold_feature_cache ALPHAFOLD_FEATURE_CACHE] [--af2_pdb AF2_PDB]

Predict model quality and output NumPy array format.

optional arguments:
  -h, --help                  Show this help message and exit
  --input INPUT               Path to input pdb file.
  --output OUTPUT             Path to output folder.
  --method METHOD             Prediction method, can be "ensemble", "EGNN_Full", "se3_Full", "EGNN_esto9" or "EGNN_covariance". Ensemble can be done listing multiple models separated by comma.
  --alphafold_prediction      Path to alphafold prediction results.               
  --alphafold_feature_cache   Optional. Can cache AlphaFold features for models of the same sequence.
  --af2_pdb AF2_PDB           Optional. PDBs from AlphaFold predcition for index correction with input pdb when input PDB only contains partial sequence of the AlphaFold results.
  --cpu                       Optional. Force to use CPU.

```


### Example usages


```
python3 EnQA.py --input example/model/6KYTP/test_model.pdb --output outputs/ --method EGNN_Full --alphafold_prediction example/alphafold_prediction/6KYTP/
```

If you want to run models based on the [SE(3)-Transformer](https://arxiv.org/abs/2006.10503), then the Python package `equivariant_attention` is required and should be installed following [Fabian's implementation](https://github.com/FabianFuchsML/se3-transformer-public).

Example:

```
python3 EnQA.py --input example/model/6KYTP/test_model.pdb --output outputs/ --method se3_Full --alphafold_prediction example/alphafold_prediction/6KYTP/  
```

### Generating AlphaFold2 models for assisted quality assessment

For generating models using AlphaFold2, an installation of AlphaFold2 following its [Official Repo](https://github.com/deepmind/alphafold) is required. For our experiments, we use its original model used at CASP14 with no ensembling (--model_preset=monomer), with all genetic databases used at CASP14 (--db_preset=full_dbs), and restricts templates only to structures that were available at the start of CASP14 (--max_template_date=2020-05-14).



### Model training for EnQA assisted with AlphaFold2
First generate 5 AlphaFold reference models 
To generate the labels and features when you have the predicted results from AlphaFold and the corresponding native PDBs, using the following procedure:
```
python3 process.py --input example/model/6KYTP/test_model.pdb --label_pdb example/label/6KYTP.pdb --output outputs/processed --alphafold_prediction example/alphafold_prediction/6KYTP/
 ```
 
Code in train.py provides a basic framework to train the EGNN_full model with Pytorch, After all feature files for training and validation are generated, suppose the processed features files(in .pt format) are saved in path/to/train/ and path/to/validation/, here is an example to train the model:
```
python3 train.py --train path/to/train/ --validation path/to/validation --output outputs/ --epochs 60
```

## Geometric feature generation

The featurizers from Spherical graph convolutional networks (S-GCN) are used to process 3D models of proteins represented as molecular graphs.
Here we provide the voronota and spherical harmonics featurizer for Linux.

If you need to rebuild the voronota for a different system, please check out the [S-GCN Repo](https://gitlab.inria.fr/GruLab/s-gcn/-/tree/master/#voronota).

Also, there are [binaries](https://gitlab.inria.fr/GruLab/s-gcn/-/tree/master/#spherical-harmonics-featurizer) built for featurizer under a different system. (Currently, only MacOS and Linux are supported)


## PDB with multiple chains
For EnQA-MSA, you can preprocess the input PDB with the [mergePDB](https://github.com/BioinfoMachineLearning/EnQA/blob/25c1142fa8936ebb843db79a51161cdee499697a/data/process_alphafold.py#L139) function we provided to convert it into a "merged single chain PDB" and make that as the input PDB.

For EnQA assisted with AlphaFold2, you can provide protein complexes as input, and no additional work is required.

## Reference
[Chen C, Chen X, Morehead A, Wu T, Cheng J. 3D-equivariant graph neural networks for protein model quality assessment. Bioinformatics. 2023 Jan 13:btad030. doi: 10.1093/bioinformatics/btad030.](https://pubmed.ncbi.nlm.nih.gov/36637199/)

