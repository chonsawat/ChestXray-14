import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

from modules.dataset import Dataset
from modules.models import Model

# Constant variables
NAME = "DenseNet121"
EPOCHS = 10
NUM_FOLDS = 5

dataset = Dataset()
for fold_num in range(1, NUM_FOLDS + 1):
    # WandbCallback
    run = wandb.init(project="Experiment 2",
                     name=f"{NAME} using fold {fold_num} as test dataset",
                     reinit=True)
    
    # Dataset
    train_dataset, test_dataset = dataset.get_kfold(fold_num, sample=True)
    
    # Modeling
    transfer_model = tf.keras.applications.densenet.DenseNet121(
        include_top=False, 
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling=None
    )
    model = Model(transfer_model).get_model(flatten=True)
    model.summary()
    
    # Visualize
    history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=test_dataset,
            verbose=1,
            callbacks=[WandbCallback()]
    )
    break
