import sys

# ----------------------Common Hyperparams-------------------------- #
num_class = 1
mlp_neurons = 128
hid_dim = 512

# ----------------------Baseline Hyperparams-------------------------- #
if '--type' in sys.argv:
    idx = sys.argv.index('--type')
    mode = sys.argv[idx+1]
else:
    mode = 'baseline'  # 기본값


#waterbirds
#"""
if mode == 'baseline':
    base_epochs     = 100
    base_batch_size = 256
    base_lr         = 0.0001
    weight_decay    = 1
    scale = 8
    std = 0.2
    K = 6
    opt_b           = 'adam'
    opt_m           = 'adam'
elif mode == 'margin':
    base_epochs     = 100
    base_batch_size = 64
    base_lr         = 0.01
    weight_decay    = 0.01
    scale = 8
    std = 0.2
    K = 6
    opt_b           = 'adam'
    opt_m           = 'adam'
#"""



#celebA
"""
if mode == 'baseline':
    base_epochs     = 50
    base_batch_size = 512
    base_lr         = 0.01
    weight_decay    = 0.5
    scale = 8
    std = 0.2
    K = 6
    opt_b           = 'sgd'
    opt_m           = 'sgd'
elif mode == 'margin':
    base_epochs     = 100
    base_batch_size = 128
    base_lr         = 0.001
    weight_decay    = 0.0001
    scale = 8
    std = 0.2
    K = 4
    opt_b           = 'sgd'
    opt_m           = 'sgd'
"""


# ----------------------Paths-------------------------- #
basemodel_path = 'basemodel.pth'
margin_path = 'margin.pth'

# ----------------------Model-details-------------------------- #
model_name = 'resnet18'

# ----------------------ImageNet Means and Transforms---------- #
imagenet_mean = [0.485, 0.456, 0.406] 
imagenet_std = [0.229, 0.224, 0.225]

# -----------------------CelebA/Waterbirds-parameters--------- #
dataset_path = './datasets'

"""
img_dir = './datasets/CelebA/img/img_align_celeba'
partition_path = './datasets/CelebA/Eval/list_eval_partition.txt'
attr_path = './datasets/CelebA/Anno/list_attr_celeba.txt'
"""
img_dir = './datasets/waterbirds/waterbird_complete95_forest2water2'
metadata_path = './datasets/waterbirds/waterbird_complete95_forest2water2/metadata.csv'


#target_attribute = 'Blond_Hair'
#bias_attribute = 'Male'

target_attribute = 'y'         # Bird type: 0 = landbird, 1 = waterbird
bias_attribute = 'place'      # Background: 0 = land, 1 = water


celeba_path = './datasets/celeba_features'
celeba_val_path = './datasets/celeba_features'
waterbirds_path = './datasets/waterbirds_features'
waterbirds_val_path = './datasets/waterbirds_features'
