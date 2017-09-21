# Configuration File
# taken from https://github.com/meliketoy/fine-tuning.pytorch/blob/master/config.py
# Base directory for data formats
#name = 'GURO_CELL'
#name = 'INBREAST'
name = 'FINAL_TRAIN_NEW_AUG'


#train_data_dir = '../data/scene_classification/scene_train_images_20170904'
#validation_data_dir = '../data/scene_classification/scene_validation_images_20170908'

data_base = '/home/mnt/datasets/'+name
aug_base = '/home/bumsoo/Data/split/'+name
#test_dir = '/home/bumsoo/Data/test/FINAL_TEST'
test_dir = '../data/scene_classification/scene_train_images_20170904'

# model option
batch_size = 16
num_epochs = 50
lr_decay_epoch=10
feature_size = 100

# meanstd options
# INBREAST
#mean = [0.60335361908536667, 0.60335361908536667, 0.60335361908536667]
#std = [0.075116530817055119, 0.075116530817055119, 0.075116530817055119]

# GURO_EXTEND
#mean = [0.48359630772217554, 0.48359630772217554, 0.48359630772217554]
#std = [0.13613821516980551, 0.13613821516980551, 0.13613821516980551]

# GURO+INBREAST
mean = [0.51508365254458033, 0.51508365254458033, 0.51508365254458033]
std = [0.12719534902225299, 0.12719534902225299, 0.12719534902225299]
