# PDB files for training and testing on AlphaFold2 predictions datset.

As the feature data is extremely large (can reach over ~1GB for a single sequence) and final PDB outputs are not sufficient for training, it would be more applicable for users to generate such features using the scripts provide in the EnQA repo.
To generate the required data for training, we provide the [fasta format sequence](https://github.com/BioinfoMachineLearning/EnQA/blob/main/data/seqs.tar.gz) for our training data and they can be used as input with the following command by [AlphaFold2](https://github.com/deepmind/alphafold):
 
 ```
 python3 docker/run_docker.py \
  --fasta_paths=T1050.fasta \
  --max_template_date=2020-05-14 \
  --model_preset=monomer \
  --db_preset=full_dbs \
  --data_dir=$DOWNLOAD_DIR
 ```


We also provide the [native PDBs](https://drive.google.com/file/d/1H7AI2cYqP5nZYhmNJULxuzrODh6elrYz/view?usp=sharing) which are required for generating the labels.

To generate the labels and features when you have the predicted results from AlphaFold and the corresponding native PDBs, run [process.py](https://github.com/BioinfoMachineLearning/EnQA/blob/main/process.py) with the following command:

```
python3 process.py --input example/model/6KYTP/test_model.pdb --label_pdb example/label/6KYTP.pdb --output outputs/processed --alphafold_prediction example/alphafold_prediction/6KYTP/
```

Code in [train.py](https://github.com/BioinfoMachineLearning/EnQA/blob/main/train.py) now provides a basic framework to train the EGNN_full model with Pytorch, After all feature files for training and validation are generated, suppose the processed features files(in .pt format) are saved in path/to/train/ and path/to/validation/, here is an example to train the model:

```
python3 train.py --train path/to/train/ --validation path/to/validation/ --output outputs/ --epochs 60
```

The training script can be modified to include the callbacks, logging and monitoring for the training procedure fit for user’s environment. Note that while EnQA can be called on PDBs with multiple chains, the training was performed in monomer datasets, and support for complexes would be added in the future.
