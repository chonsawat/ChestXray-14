import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

from modules.dataset import Dataset
from modules.models import Model

# Constant variables
NAME = "ResNet50"
EPOCHS = 10
NUM_FOLDS = 5

dataset = Dataset()
for fold_num in range(1, NUM_FOLDS + 1):
    # WandbCallback
    run = wandb.init(project="Experiment 1",
                    name=f"{NAME} using fold {fold_num} as test dataset")
    
    # Dataset
    train_dataset, test_dataset = dataset.get_kfold(fold_num)
    
    # Modeling
    transfer_model = tf.keras.applications.resnet50.ResNet50(
        include_top=True, 
        weights=None,
        pooling='avg'
    )
    model = Model(transfer_model).get_model()
    
    # Visualize
    history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=test_dataset,
            verbose=1,
            callbacks=[WandbCallback()])
