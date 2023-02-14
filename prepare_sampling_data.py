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
        
df = pd.read_csv(f"{INPUT_PATH}/under_sampling_data.csv", index_col=0)
df = df.astype("int16")
df = shuffle(df, random_state=SEED)

NUM_FILES = 112
Directory().create_sampling_data_folder()

tfrec_path = f'{INPUT_PATH}/data/sampling'
for i, fold in tqdm(enumerate(np.array_split(df, NUM_FILES)), total=NUM_FILES):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')

print("Done Preparing!!")