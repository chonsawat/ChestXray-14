""" Documations
Modules for parepare dataset
"""

import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')


BATCH_SIZE = 16
IMG_SIZE = 224
SEED = 42

feature_map = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_id': tf.io.FixedLenFeature([], tf.string),
    'No Finding': tf.io.FixedLenFeature([], tf.int64)
}


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
        example['No Finding']]
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
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
    if cached:
        dataset = dataset.cache()
    dataset = dataset.prefetch(auto)
    return dataset


# Create TFRecordDataset
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
        'No Finding': tf.train.Feature(int64_list=tf.train.Int64List(value=[proba[0]]))}
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
        
def serialize_sampling(data, name):
    samples = []
    
    for index, proba in data.iterrows():
        # print(index)
        samples.append(_serialize_sample(
            index.split('/')[-1].encode(), 
            _serialize_image(index), 
            proba))
    
    with tf.io.TFRecordWriter(name + '.tfrec') as writer:
        [writer.write(x) for x in samples]