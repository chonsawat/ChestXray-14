""" Documations
Path:
    Elab Path: 
        input_path = "~/ChestXray-14/dataset/ChestXray NIH"
"""

from calendar import EPOCH
import wandb
import tensorflow as tf
from wandb.keras import WandbCallback

from modules.utils import *
from modules.models import Model
from modules.dataset import Dataset

# Constant variables
NAME = "ResNet50"
EPOCHS = 10
input_path = "dataset/ChestXray NIH"

# Wandb
run = wandb.init(project="Experiment 1",
                 name=NAME)

# Dataset
dataset = Dataset()
train_dataset = dataset.get_train()
val_dataset = dataset.get_valid()

# Modeling
transfer_model = tf.keras.applications.resnet50.ResNet50(
    include_top=True, 
    weights=None,
    pooling='avg'
)
model = Model(transfer_model).get_model()

# Visualize
history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        verbose=1,
        callbacks=[WandbCallback()])