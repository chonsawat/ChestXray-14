from sklearn.utils import shuffle
from tqdm.notebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
from modules.dataset import Directory

INPUT_PATH = "dataset/ChestXray NIH"
STRATEGY = tf.distribute.get_strategy()    
BATCH_SIZE = 16
IMG_SIZE = 224
SEED = 42

def _serialize_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.uint8)
    return tf.image.encode_jpeg(image).numpy()


def _serialize_sample(image_id, image, proba):
    feature = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'image_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_id])),
        'No Finding': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[0]])),
        'Atelectasis': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[1]])),
        'Consolidation': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[2]])),
        'Infiltration': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[3]])),
        'Pneumothorax': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[4]])),
        'Edema': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[5]])),
        'Emphysema': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[6]])),
        'Fibrosis': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[7]])),
        'Effusion': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[8]])),
        'Pneumonia': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[9]])),
        'Pleural_Thickening': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[10]])),
        'Cardiomegaly': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[11]])),
        'Nodule': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[12]])),
        'Mass': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[13]])),
        'Hernia': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[14]]))}
    sample = tf.train.Example(features=tf.train.Features(feature=feature))
    return sample.SerializeToString()


def serialize_fold(fold, name):
    samples = []
    
    for index, proba in fold.iterrows():
        samples.append(_serialize_sample(
            index.split('/')[-1].encode(), 
            _serialize_image(index), 
            proba))
    
    with tf.io.TFRecordWriter(name + '.tfrec') as writer:
        [writer.write(x) for x in samples]
        
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
