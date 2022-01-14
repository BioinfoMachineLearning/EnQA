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
```

# Example usages

Running on a model under example folder
```
python3 predict.py --input example/model/6KYTP/test_model.pdb --output outputs/prediction/ --model_path models/egnn_ne.tar --disto_type esto9 --model_type egnn_ne --alphafold_prediction example/alphafold_prediction/6KYTP/
```

If you want to run models based on [SE(3)-Transformer](https://arxiv.org/abs/2006.10503), then Python package equivariant_attention is required, and should be installed following [Fibian's implementation](https://github.com/FabianFuchsML/se3-transformer-public).