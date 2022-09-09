from sklearn.utils import shuffle
from tqdm.notebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import os

input_path = "dataset/ChestXray NIH"
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
        
df = pd.read_csv(f"{input_path}/preprocessed_data.csv", index_col=0)
df = df.astype("int16")
df = shuffle(df, random_state=SEED)
df_train = df[:78484] # 70% : 78484 : 179 folds
df_valid = df[78484:100908] # 20% : 22,424 : 51 fold
df_test = df[100908:] # 10% : 11,212 : 25 fold

folds = 256

tfrec_path = f'{input_path}/data/224x224/train'
for i, fold in tqdm(enumerate(np.array_split(df_train, folds)), total=folds):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')
    
tfrec_path = f'{input_path}/data/224x224/valid'
for i, fold in tqdm(enumerate(np.array_split(df_valid, folds)), total=folds):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')

tfrec_path = f'{input_path}/data/224x224/test'
for i, fold in tqdm(enumerate(np.array_split(df_test, folds)), total=folds):
    serialize_fold(fold, name=f'{tfrec_path}/{i:03d}-{len(fold):03d}')