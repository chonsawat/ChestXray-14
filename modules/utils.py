""" Documations
Modules for parepare dataset
"""

import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')


STRATEGY = tf.distribute.get_strategy()    
BATCH_SIZE = 16
IMG_SIZE = 224
SEED = 42


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


def decode_image_color(image_data):
    """Decode Image as RGB Scale

    Parameters
    ----------
    image_data : np.arrays
        image data in Gray scale format

    Returns
    -------
    image
        image data in RGB scale format
    """    
    image = tf.image.decode_jpeg(image_data, channels=1)
    image = tf.image.grayscale_to_rgb(image)
    image = tf.reshape(image, [IMG_SIZE, IMG_SIZE, 3])
    return image


def scale_image(image, target):
    """Scaling image

    Parameters
    ----------
    image : np.arrays
        a image data
    target : np.arrays
        a labels of that image

    Returns
    -------
    tuple
        tuple of (image, target)
    """
    image = tf.cast(image, tf.float32) / 255.
    return image, target


def read_tfrecord_color(example):
    """Read binary data & Convert to RGB dataset

    Parameters
    ----------
    example : tf.Tensor
        Tensor of data and labels

    Returns
    -------
    tf.Tensor
        Tensor of data and labels
    """    
    example = tf.io.parse_single_example(example, feature_map)
    image = decode_image_color(example['image'])
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


def get_dataset(filenames, cached=False):
    """get dataset form tfrec filenames & convert to tf.Tensor: (image, target)

    Parameters
    ----------
    filenames : list
        list of filenames from *.tfrec
    cached : bool, optional
        Keep cache in memory, by default False

    Returns
    -------
    tf.Tensor
        a tf.Dataset for ready to use
    """    
    auto = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=auto)
    dataset = dataset.map(read_tfrecord_color, num_parallel_calls=auto)
        
    dataset = dataset.map(scale_image, num_parallel_calls=auto)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    if cached:
        dataset = dataset.cache()
    dataset = dataset.prefetch(auto)
        
    print(dataset)
    return dataset
