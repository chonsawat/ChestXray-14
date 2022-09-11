from sklearn.utils import shuffle
from tqdm.notebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
from modules.dataset import Directory
from modules.utils import serialize_fold

INPUT_PATH = "dataset/ChestXray NIH"
STRATEGY = tf.distribute.get_strategy()    
BATCH_SIZE = 16
IMG_SIZE = 224
SEED = 42
        
df = pd.read_csv(f"{INPUT_PATH}/preprocessed_data.csv", index_col=0)
df = df.astype("int16")
df = shuffle(df, random_state=SEED)
df = df.head(1000)
df1, df2, df3, df4, df5 = np.split(df, 5)

FOLDS = 8
Directory().create_sample_folds_folder()

# Fold1
tfrec_path = f'{INPUT_PATH}/data/sample_folds/fold1'
for i, fold in tqdm(enumerate(np.array_split(df1, FOLDS)), total=FOLDS):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')

# Fold2
tfrec_path = f'{INPUT_PATH}/data/sample_folds/fold2'
for i, fold in tqdm(enumerate(np.array_split(df2, FOLDS)), total=FOLDS):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')

# Fold3
tfrec_path = f'{INPUT_PATH}/data/sample_folds/fold3'
for i, fold in tqdm(enumerate(np.array_split(df3, FOLDS)), total=FOLDS):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')

# Fold4
tfrec_path = f'{INPUT_PATH}/data/sample_folds/fold4'
for i, fold in tqdm(enumerate(np.array_split(df4, FOLDS)), total=FOLDS):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')

# Fold5
tfrec_path = f'{INPUT_PATH}/data/sample_folds/fold5'
for i, fold in tqdm(enumerate(np.array_split(df5, FOLDS)), total=FOLDS):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')
    
print("Done Preparing!!")
