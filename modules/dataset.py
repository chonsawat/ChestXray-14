import os
import tensorflow as tf
if __name__ == '__main__': 
    from utils import get_dataset
else: 
    from modules.utils import get_dataset

INPUT_PATH = "/home/jovyan/ChestXray-14/dataset/ChestXray NIH"
LABELS = ['No Finding', 'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

class Dataset:
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
    
    def get_kfold(self, fold_number:int, sample=False):
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
            if sample:
                filenames = tf.io.gfile.glob(f'{INPUT_PATH}/data/sample_folds/{folder}/*.tfrec')
            else:
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
        
    def create_sampling_data_folder(self):
        os.makedirs(os.path.join(INPUT_PATH, "data", "sampling"), exist_ok=True)

if __name__ == '__main__':
    Directory().create_folds_folder()
    Directory().create_sampling_data_folder()
    print("Done")
