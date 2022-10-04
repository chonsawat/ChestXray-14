import os

import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from modules.dataset import Dataset
from modules.models import Model
from modules.parser import parse_option

option = parse_option()
weight_option = 'imagenet' if option.imagenet else None

# Constant variables
NAME = "Resnet50"
EPOCHS = 100
NUM_FOLDS = 5

# Learning rate
def lr_schedule(epoch, learning_rate):
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

# Model Checkpoint
model_checkpoint_callback = ModelCheckpoint(f'results/models/{NAME}.h5', monitor='val_loss', mode='min', save_best_only=True)
early_stop_callback = EarlyStopping(monitor='val_loss', mode="min", patience=20, verbose=1)
reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', mode="min", factor=0.5, patience=5, verbose=1)
lr_logging_callback = LearningRateScheduler(lr_schedule)

dataset = Dataset()
for fold_num in range(1, NUM_FOLDS + 1):
    # WandbCallback
    run = wandb.init(project="Experiment 3",
                     name=f"{NAME} using fold {fold_num} as test dataset (weight={weight_option}) - Complete",
                     reinit=True)
    # CSV Logger
    path = os.path.join("results", "history", NAME)
    os.makedirs(path, exist_ok=True)
    csv_logger = CSVLogger(os.path.join(path, f"fold_{fold_num}.csv"))
    
    # Dataset
    train_dataset, test_dataset = dataset.get_kfold(fold_num, sample=False)
    
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
        callbacks=[WandbCallback(), model_checkpoint_callback, csv_logger, early_stop_callback, reduce_lr_callback, lr_logging_callback],
        # callbacks=[model_checkpoint_callback, csv_logger, early_stop_callback, reduce_lr_callback, lr_logging_callback]
    )
    
    # break