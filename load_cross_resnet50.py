import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

from modules.dataset import Dataset
from modules.models import Model

# Constant variables
NAME = "ResNet50"
EPOCHS = 10
NUM_FOLDS = 5

dataset = Dataset()
fold_num = 1

# Dataset
train_dataset, test_dataset = dataset.get_kfold(fold_num, sample=True)

# Modeling
transfer_model = tf.keras.applications.resnet50.ResNet50(
    include_top=True, 
    weights=None,
    pooling='avg'
)
model = Model(transfer_model).get_model()

# Visualize
best_model = wandb.restore('model-best.h5', run_path="chestxray/Experiment 2/9rkc5pvu")
model.load_weights(best_model.name)
    
model.save("results/models/resnet50_fold1.h5")