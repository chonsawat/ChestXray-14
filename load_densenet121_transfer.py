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
def get_densenet121_model():
    """
    get_densenet121_model():
        create the tensorflow model
        ===========================
        return:
            pretrained-model restnet50
            
            
        Example use
        ===========
        >>> model = get_densenet121_model()
        
    """
    model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(
            include_top=False, 
            input_shape=(None, None, 3),  # new dataset is grey-scale image
            weights='imagenet',
            pooling='avg'
        ),
        tf.keras.layers.Dense(15, activation='sigmoid')  # 15 Output for new datasets
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
        model = get_densenet121_model()

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=tf.keras.metrics.AUC(multi_label=True))
        
        # TODO: Restore model
        best_model = wandb.restore('model-best.h5', run_path='chestxray/ChestXray/3mh42m31')
        model.load_weights(best_model.name)


    model.save(f"/home/jovyan/ChestXray-14/results/models/Densenet121_Transfer_epochs-20.h5")
else:
    print("\n===== Please, install GPU =====")
# ====================================================================