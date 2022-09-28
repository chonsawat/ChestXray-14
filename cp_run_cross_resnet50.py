import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import ModelCheckpoint

from modules.dataset import Dataset
from modules.models import Model
from modules.parser import parse_option

option = parse_option()
weight_option = 'imagenet' if option.imagenet else None

# Constant variables
NAME = "Resnet50"
EPOCHS = 10
NUM_FOLDS = 5

# Model Checkpoint
mc = ModelCheckpoint('results/models/best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

dataset = Dataset()
for fold_num in range(1, NUM_FOLDS + 1):
    # WandbCallback
    run = wandb.init(project="Experiment 3",
                     name=f"{NAME} using fold {fold_num} as test dataset (weight={weight_option}) _ check point",
                     reinit=True)
    
    # Dataset
    train_dataset, test_dataset = dataset.get_kfold(fold_num, sample=True)
    
    # Modeling
    transfer_model = tf.keras.applications.resnet50.ResNet50(
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
            # callbacks=[WandbCallback()],
            callbacks=[WandbCallback(), mc]
    )
    
    break