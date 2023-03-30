import os
import tensorflow as tf
if __name__ == '__main__': 
    from utils import get_dataset
else: 
    from modules.utils import get_dataset

ROOT_PATH = "/home/jovyan/ChestXray-14"
INPUT_PATH = f"{ROOT_PATH}/dataset/ChestXray NIH"
LABELS = ['No Finding']

class Dataset:
    
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
        experiment_dataset_path = "data/binary_dataset"
        folders = os.listdir(f"{INPUT_PATH}/{experiment_dataset_path}/folds")
        assert \
            fold_number > 0 and fold_number <= len(folders), \
            "Error: fold_number must be between 0 and len(fold_folders)"
        
        train_filenames = []
        test_filenames = []
        for folder in folders:

            filenames = tf.io.gfile.glob(f'{INPUT_PATH}/{experiment_dataset_path}/folds/{folder}/*.tfrec')

            if folder == f"fold{fold_number}":
                test_filenames = test_filenames + filenames
            else:
                train_filenames = train_filenames + filenames
                
        train_dataset = get_dataset(train_filenames)
        test_dataset = get_dataset(test_filenames)
        return train_dataset, test_dataset