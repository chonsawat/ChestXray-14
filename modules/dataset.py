import os
import tensorflow as tf
if __name__ == '__main__': 
    from utils import get_dataset
else: 
    from modules.utils import get_dataset


class Dataset:
    INPUT_PATH = "dataset/ChestXray NIH"

    def get_train(self):
        filenames = tf.io.gfile.glob(f'{self.INPUT_PATH}/data/224x224/train/*.tfrec')
        dataset = get_dataset(filenames)
        return dataset

    def get_test(self):
        filenames = tf.io.gfile.glob(f'{self.INPUT_PATH}/data/224x224/test/*.tfrec')
        dataset = get_dataset(filenames)
        return dataset

    def get_valid(self):
        filenames = tf.io.gfile.glob(f'{self.INPUT_PATH}/data/224x224/valid/*.tfrec')
        dataset = get_dataset(filenames)
        return dataset

    def get_all(self):
        filenames = tf.io.gfile.glob(f'{self.INPUT_PATH}/data/224x224/All/*.tfrec')
        dataset = get_dataset(filenames)
        return dataset
    
    def get_sample(self):
        filenames = tf.io.gfile.glob(f'{self.INPUT_PATH}/data/sample/*.tfrec')
        dataset = get_dataset(filenames)
        return dataset
    
    def get_kfold(self, fold_number:int):
        """Get dataset for given fold number as test dataset

        Parameters
        ----------
        fold_number : int
            number of fold to get as test dataset

        Returns
        -------
        tuple
            tuple of (train, test) datasets
        """        
        folders = os.listdir(f"{INPUT_PATH}/data/folds")
        assert \
            fold_number > 0 and fold_number <= len(folders), \
            "Error: fold_number must be between 0 and len(fold_folders)"
        
        train_filenames = []
        test_filenames = []
        for folder in folders:
            filenames = tf.io.gfile.glob(f'{INPUT_PATH}/data/folds/{folder}/*.tfrec')
            if folder == f"fold{fold_number}":
                test_filenames = test_filenames + filenames
            else:
                train_filenames = train_filenames + filenames
                
        train_dataset = get_dataset(train_filenames)
        test_dataset = get_dataset(test_filenames)
        return train_dataset, test_dataset


class Directory():
    def create_folds_folder(self):
        os.makedirs(os.path.join(INPUT_PATH, "data", "folds", "fold1"), exist_ok=True)
        os.makedirs(os.path.join(INPUT_PATH, "data", "folds", "fold2"), exist_ok=True)
        os.makedirs(os.path.join(INPUT_PATH, "data", "folds", "fold3"), exist_ok=True)
        os.makedirs(os.path.join(INPUT_PATH, "data", "folds", "fold4"), exist_ok=True)
        os.makedirs(os.path.join(INPUT_PATH, "data", "folds", "fold5"), exist_ok=True)
        
    def create_sample_folds_folder(self):
        os.makedirs(os.path.join(INPUT_PATH, "data", "sample_folds", "fold1"), exist_ok=True)
        os.makedirs(os.path.join(INPUT_PATH, "data", "sample_folds", "fold2"), exist_ok=True)
        os.makedirs(os.path.join(INPUT_PATH, "data", "sample_folds", "fold3"), exist_ok=True)
        os.makedirs(os.path.join(INPUT_PATH, "data", "sample_folds", "fold4"), exist_ok=True)
        os.makedirs(os.path.join(INPUT_PATH, "data", "sample_folds", "fold5"), exist_ok=True)

if __name__ == '__main__':
    INPUT_PATH = "dataset/ChestXray NIH"
    Directory().create_folds_folder()
    print("Done")