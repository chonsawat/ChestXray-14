import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

from modules.dataset import Dataset
from modules.models import Model

option = parse_option()
weight_option = 'imagenet' if option.imagenet else None

# Constant variables
NAME = "EfficientNetB0"
EPOCHS = 10
NUM_FOLDS = 5

dataset = Dataset()
for fold_num in range(1, NUM_FOLDS + 1):
    # WandbCallback
    run = wandb.init(project="Experiment 3",
                     name=f"{NAME} using fold {fold_num} as test dataset (weight=imagenet)",
                     reinit=True)
    
    # Dataset
    train_dataset, test_dataset = dataset.get_kfold(fold_num, sample=False)
    
    # Modeling
    transfer_model = tf.keras.applications.efficientnet.EfficientNetB0(
        include_top=False, 
        weights=weight_option,
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
