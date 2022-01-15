# EnQA
A 3D-equivariant neural network for protein structure accuracy estimation


```
usage: predict.py [-h] --input INPUT --output OUTPUT --model_path MODEL_PATH --disto_type DISTO_TYPE [--model_type MODEL_TYPE] [--alphafold_prediction ALPHAFOLD_PREDICTION]
                  [--alphafold_feature_cache ALPHAFOLD_FEATURE_CACHE]

Error predictor network

positional arguments:
  input                 path to input pdb file
  output                path to output (can be path or filename)

optional arguments:
  -h, --help                    show this help message and exit
  --input                       path to input pdb file
  --output                      path to output (can be path or filename)
  --model_path                  path to model weight file
  --disto_type                  type of 2D features, can be "cov64", "cov25" or "esto9"
  --alphafold_prediction        path to AlphaFold2 predictions, should include files with names: result_model_[1-5].pkl
  --alphafold_feature_cache     Optional, path to temp folder which saves intermediate alphafold_features
  --af2_pdb                     Optional. PDBs from AlphaFold2 predcition. Used for index correction when missing residues exist in input file.
```

# Example usages

Running on a E(n)-Equivariant model under example folder:

```
python3 predict.py --input example/model/6KYTP/test_model.pdb --output outputs/prediction/ --model_path models/egnn_ne.tar --disto_type esto9 --model_type egnn_ne --alphafold_prediction example/alphafold_prediction/6KYTP/
```

If you want to run models based on [SE(3)-Transformer](https://arxiv.org/abs/2006.10503), then Python package equivariant_attention is required, and should be installed following [Fibian's implementation](https://github.com/FabianFuchsML/se3-transformer-public).

Example:

```
python3 predict.py --input example/model/6KYTP/test_model.pdb --output outputs/prediction/ --model_path models/esto9_se3.tar --disto_type se3 --model_type egnn_ne --alphafold_prediction example/alphafold_prediction/6KYTP/
```

# Generating AlphaFold2 models for assisted quality assessment

For generating models using AlphaFold2, an installation of AlphaFold2 following its [Official Repo](https://github.com/deepmind/alphafold) is required. For our experiments, we use its original model used at CASP14 with no ensembling (--model_preset=monomer), with all genetic databases used at CASP14 (--db_preset=full_dbs), and restricts templates only to structures that were available at the start of CASP14 (--max_template_date=2020-05-14).
