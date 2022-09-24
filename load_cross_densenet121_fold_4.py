import os

import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

from modules.dataset import Dataset
from modules.models import Model

# Constant variables
NAME = "DenseNet121"

# Modeling
transfer_model = tf.keras.applications.densenet.DenseNet121(
    include_top=False, 
    weights='imagenet',
    input_shape=(224, 224, 3),
    pooling=None
)
model = Model(transfer_model).get_model(flatten=True)

# Visualize
best_model = wandb.restore('model-best.h5', run_path="chestxray/Experiment 3/lvew4cw6")
model.load_weights(best_model.name)
    
model.save(f"results/models/{NAME}_fold_4.h5")
os.system("rm model-best.h5")
