import tensorflow as tf
from modules.utils import get_dataset


class Dataset:
    input_path = "dataset/ChestXray NIH"

    def get_train(self):
        filenames = tf.io.gfile.glob(f'{self.input_path}/data/224x224/train/*.tfrec')
        dataset = get_dataset(filenames)
        return dataset

    def get_test(self):
        filenames = tf.io.gfile.glob(f'{self.input_path}/data/224x224/test/*.tfrec')
        dataset = get_dataset(filenames)
        return dataset

    def get_valid(self):
        filenames = tf.io.gfile.glob(f'{self.input_path}/data/224x224/valid/*.tfrec')
        dataset = get_dataset(filenames)
        return dataset

    def get_all(self):
        filenames = tf.io.gfile.glob(f'{self.input_path}/data/224x224/All/*.tfrec')
        dataset = get_dataset(filenames)
        return dataset