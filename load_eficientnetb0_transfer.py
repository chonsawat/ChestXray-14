""" Documations
Path:
    Elab Dataset Path: 
        input_path = "~/ChestXray-14/dataset/ChestXray NIH"
"""
# Modules
from sklearn.utils import shuffle
from tqdm.notebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import os

# ChestXray Required Modules
from utils import *

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
            input_shape=(None, None, 3),
            weights='imagenet',
            pooling='avg'),
        tf.keras.layers.Dense(15, activation='sigmoid')
    ])
    return model


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

    test_filenames = tf.io.gfile.glob(f'{input_path}/data/224x224/test/*.tfrec')

    with STRATEGY.scope():
        tf.keras.backend.clear_session()
        model = get_model()

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=tf.keras.metrics.AUC(multi_label=True))
        
        # TODO: Restore model
        best_model = wandb.restore('model-best.h5', run_path='chestxray/ChestXray/1evo7bd3')
        model.load_weights(best_model.name)


    model.save(f"/home/jovyan/ChestXray-14/results/models/EfficientNetB0_transfer_epochs-20.h5")
    print("Saved")
    os.system("rm /home/jovyan/ChestXray-14/model-best.h5")
else:
    print("\n===== Please, install GPU =====")
# ====================================================================