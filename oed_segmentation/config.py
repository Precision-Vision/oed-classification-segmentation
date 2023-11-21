# Paths
dataset_path_batch_01 = '/home/u2271662/tia/projects/pv/data/batch_01_rev01/'
dataset_path_batch_02 = '/home/u2271662/tia/projects/pv/data/batch_02_rev02/'
dataset_path_test_01 = '/home/u2271662/tia/projects/pv/data/external/chandrashekar/'
dataset_path_test_02 = '/home/u2271662/tia/projects/pv/data/external/barot/'
model_path = '/home/u2271662/tia/projects/pv/code/pv/maskrcnn-torch/models/'

# CONFIG
batch_no = 2
dataset_path = dataset_path_batch_01 if batch_no == 1 else dataset_path_batch_02
linecolors = ['65280', '10485760'] if batch_no == 1 else ['255', '16711680']
model_filename = 'model22-25ep-batch-02-2-classes.pt' # 'model18-50ep-batch-02.pt' # 'model12-50ep-aug-3-1-class-maskrcnnv2.pt' # 'model.pt'
PRE_PLOT = True
PRED_PLOT = True
TRAIN_MODE = False # If False, will load model from model_filename
fig_size = [10, 2.5] # [15, 5]
num_epochs = 25
num_classes = 2
downsample = 3
minimum_size = [512, 512]
one_class_mode = False if num_classes > 2 else True
random_val = True
learning_rate = 0.0001
weight_decay = 0.0005
train_split_ratio = [0.80, 0.20]
overlap_threshold = 0.25
max_detections = 1
eval_times = 5
excluded_classes = [] if batch_no == 2 else []