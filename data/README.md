# PDB files for training and testing on AlphaFold2 predictions datset.

 The PDB files can be downloaded using the following links:

 [Experimental validated PDBs](https://drive.google.com/file/d/1H7AI2cYqP5nZYhmNJULxuzrODh6elrYz/view?usp=sharing)

 AlphaFold2 predicted PDBs are not included since the representatations are extremely large (can be over ~1GB for a single sequence) and final PDB outputs are not sufficient for training. To generat the required data for training, we provide the fasta sequence for our training data and they can be used as input with the follwing command by [AlphaFold2](https://github.com/deepmind/alphafold):
 
 ```
 python3 docker/run_docker.py \
  --fasta_paths=T1050.fasta \
  --max_template_date=2020-05-14 \
  --model_preset=monomer \
  --db_preset=full_dbs \
  --data_dir=$DOWNLOAD_DIR
 ```
 
We use 5 models generated for each target, the filename format is [Target ID].relaxed_model_[1-5].pdb 
