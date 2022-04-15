from sklearn.utils import shuffle
from tqdm.notebook import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import os


feature_map = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_id': tf.io.FixedLenFeature([], tf.string),
    'No Finding': tf.io.FixedLenFeature([], tf.int64),
    'Atelectasis': tf.io.FixedLenFeature([], tf.int64),
    'Consolidation': tf.io.FixedLenFeature([], tf.int64),
    'Infiltration': tf.io.FixedLenFeature([], tf.int64),
    'Pneumothorax': tf.io.FixedLenFeature([], tf.int64),
    'Edema': tf.io.FixedLenFeature([], tf.int64),
    'Emphysema': tf.io.FixedLenFeature([], tf.int64),
    'Fibrosis': tf.io.FixedLenFeature([], tf.int64),
    'Effusion': tf.io.FixedLenFeature([], tf.int64),
    'Pneumonia': tf.io.FixedLenFeature([], tf.int64),
    'Pleural_Thickening': tf.io.FixedLenFeature([], tf.int64),
    'Cardiomegaly': tf.io.FixedLenFeature([], tf.int64),
    'Nodule': tf.io.FixedLenFeature([], tf.int64),
    'Mass': tf.io.FixedLenFeature([], tf.int64),
    'Hernia': tf.io.FixedLenFeature([], tf.int64)}


def count_data_items(filenames):
    return np.sum([int(x[:-6].split('-')[-1]) for x in filenames])


def decode_image(image_data, IMG_SIZE=224):
    image = tf.image.decode_jpeg(image_data, channels=1)
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 1])
    return image


def scale_image(image, target):
    image = tf.cast(image, tf.float32) / 255.
    return image, target


def read_tfrecord(example):
    example = tf.io.parse_single_example(example, feature_map)
    image = decode_image(example['image'])
    target = [
        example['No Finding'],
        example['Atelectasis'],
        example['Consolidation'],
        example['Infiltration'],
        example['Pneumothorax'],
        example['Edema'],
        example['Emphysema'],
        example['Fibrosis'],
        example['Effusion'],
        example['Pneumonia'],
        example['Pleural_Thickening'],
        example['Cardiomegaly'],
        example['Nodule'],
        example['Mass'],
        example['Hernia']]
    return image, target


def data_augment(image, target, SEED=42):
    image = tf.image.random_flip_left_right(image, seed=SEED)
    image = tf.image.random_flip_up_down(image, seed=SEED)
    return image, target


def get_dataset(filenames, shuffled=False, repeated=False, 
                cached=False, augmented=False, distributed=True, STRATEGY=tf.distribute.get_strategy(), BATCH_SIZE=16, SEED=42):
    auto = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=auto)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=auto)
    if augmented:
        dataset = dataset.map(data_augment, num_parallel_calls=auto)
    dataset = dataset.map(scale_image, num_parallel_calls=auto)
    if shuffled:
        dataset = dataset.shuffle(2048, seed=SEED)
    if repeated:
        dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    if cached:
        dataset = dataset.cache()
    dataset = dataset.prefetch(auto)
    if distributed:
        dataset = STRATEGY.experimental_distribute_dataset(dataset)
    return dataset


def get_model():
    model = tf.keras.models.Sequential([
        tf.keras.applications.EfficientNetB0(
            include_top=False,
            input_shape=(None, None, 1),
            weights=None,
            pooling='avg'),
        tf.keras.layers.Dense(15, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=tf.keras.metrics.AUC(multi_label=True))

    return model

def get_resnet50_model():
    model = tf.keras.models.Sequential([
        tf.keras.applications.resnet50.ResNet50(
            include_top=False, 
            input_shape=(None, None, 1), 
            weights=None,
            pooling='avg'
            # classes=1000,
        ),
        tf.keras.layers.Dense(15, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=tf.keras.metrics.AUC(multi_label=True))

    return model