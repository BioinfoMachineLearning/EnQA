import os

from feature import create_train_dataset

model_dir = 'example/model'
target_dir = 'example/label'
alphafold_prediction_path = 'example/alphafold_prediction'

output_feature_model_path = 'outputs/train/feature_model'
output_feature_target_path = 'outputs/train/feature_target'

if not os.path.isdir(output_feature_model_path):
    os.mkdir(output_feature_model_path)

if not os.path.isdir(output_feature_target_path):
    os.mkdir(output_feature_target_path)

for target in os.listdir(model_dir):
    template_path = os.path.join(target_dir, target+'.pdb')
    load_path = os.path.join(model_dir, target)
    for model in os.listdir(load_path):
        input_model_path = os.path.join(load_path, model)
        af_path = os.path.join(alphafold_prediction_path, target)
        print('Generating feature for: {} with native structure: {}'.format(input_model_path, template_path))
        create_train_dataset(input_model_path, output_feature_model_path, output_feature_target_path, template_path,
                             diff_cutoff=15, coordinate_factor=0.01, disto_type='cov25',
                             alphafold_prediction_path=af_path, lddt_cmd='utils/lddt',
                              transform_method='rbmt', tmalign_path='utils/TMalign')
