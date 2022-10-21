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
df1, df2, df3, df4, df5 = np.split(df, 5)

NUM_FILES = 112
Directory().create_folds_folder()

# Fold1
tfrec_path = f'{INPUT_PATH}/data/folds/fold1'
for i, fold in tqdm(enumerate(np.array_split(df1, NUM_FILES)), total=NUM_FILES):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')

# Fold2
tfrec_path = f'{INPUT_PATH}/data/folds/fold2'
for i, fold in tqdm(enumerate(np.array_split(df2, NUM_FILES)), total=NUM_FILES):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')

# Fold3
tfrec_path = f'{INPUT_PATH}/data/folds/fold3'
for i, fold in tqdm(enumerate(np.array_split(df3, NUM_FILES)), total=NUM_FILES):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')

# Fold4
tfrec_path = f'{INPUT_PATH}/data/folds/fold4'
for i, fold in tqdm(enumerate(np.array_split(df4, NUM_FILES)), total=NUM_FILES):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')

# Fold5
tfrec_path = f'{INPUT_PATH}/data/folds/fold5'
for i, fold in tqdm(enumerate(np.array_split(df5, NUM_FILES)), total=NUM_FILES):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')
    
print("Done Preparing!!")
