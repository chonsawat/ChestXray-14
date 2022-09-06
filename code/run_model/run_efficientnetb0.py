# Modules
from sklearn.utils import shuffle
from tqdm.notebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# ChestXray Required Modules
from modules.utils import *

# Weight & Bias
import wandb
from wandb.keras import WandbCallback

# Close warnings messages
import warnings
warnings.filterwarnings('ignore')

# ========= Check GPU ==============
print('Using tensorflow %s' % tf.__version__)
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.config.list_physical_devices('GPU')))
else:
    print("Please install GPU version of TF")
print("\n")


# ============ Function =============
def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.applications.EfficientNetB0(
            include_top=False,
            input_shape=(None, None, 1),
            weights=None,
            pooling='avg'),
        tf.keras.layers.Dense(15, activation='sigmoid')
    ])
    return model

# =========== Weight & Bias ==============
run = wandb.init(project="ChestXray",
                config = {
                  "epochs": 20,
                  "batch_size": BATCH_SIZE,
                  "loss_function": "binary_crossentropy"
                })
config = wandb.config


# =========== Declare Variable ==============
STRATEGY = tf.distribute.get_strategy()    
BATCH_SIZE = 16
IMG_SIZE = 224
SEED = 42
input_path = "dataset/ChestXray NIH"


# ============ Main Program ======================================
if tf.test.gpu_device_name():
    """
    Check if a GPU is none it's will terminate programs.
    """
    train_filenames = tf.io.gfile.glob(f'{input_path}/data/224x224/train/*.tfrec')
    val_filenames = tf.io.gfile.glob(f'{input_path}/data/224x224/valid/*.tfrec')
    test_filenames = tf.io.gfile.glob(f'{input_path}/data/224x224/test/*.tfrec')

    train_dataset = get_dataset(train_filenames, shuffled=False, repeated=False, augmented=False)
    val_dataset = get_dataset(val_filenames, cached=True)

    with STRATEGY.scope():
        tf.keras.backend.clear_session()
        model = get_model()
        
        model.compile(
            optimizer='adam',
            loss=config.loss_function,
            metrics=tf.keras.metrics.AUC(multi_label=True))

    history = model.fit(
        train_dataset,
        epochs=config.epochs,
        # steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        # validation_steps=validation_steps,
        verbose=1,
        callbacks=[WandbCallback()])

    model.save(f"/home/jovyan/ChestXray-14/results/models/EfficientNetB0_epochs-{config.epochs}.h5")
else:
    print("\n===== Please, install GPU =====")
# ====================================================================