import os
import sys
sys.path.append('/home/jovyan/ChestXray-14')

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

from modules.dataset import Dataset
from modules.models import Model
from modules.parser import parse_option

weight_option = None # use `imagenet` or `None` only

# Constant variables
NAME = "EfficientNetB0"
EPOCHS = 100
NUM_FOLDS = 5

# Learning rate
def lr_schedule(epoch, learning_rate):
    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

def get_callbacks(NAME, weight_option, fold_num):
    model_checkpoint_callback = ModelCheckpoint(f'results/models/facal_loss/{NAME}_{weight_option}_fold_{fold_num}.h5', 
                                                monitor='val_loss', mode='min', save_best_only=True)
    early_stop_callback = EarlyStopping(monitor='val_loss', mode="min", patience=20, verbose=1)
    reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', mode="min", factor=0.5, patience=3, verbose=1)
    lr_logging_callback = LearningRateScheduler(lr_schedule)
    
    return model_checkpoint_callback, early_stop_callback, reduce_lr_callback, lr_logging_callback

dataset = Dataset()

fold_num = 3 # use values [1-5]

# Callbacks
model_checkpoint_callback, early_stop_callback, reduce_lr_callback, lr_logging_callback = get_callbacks(NAME, weight_option, fold_num)

# Path for CSV
path = os.path.join("results", "history", "training_with_facal_loss", f"{NAME}_{weight_option}")
os.makedirs(path, exist_ok=True)

# CSV Logger
csv_logger = CSVLogger(os.path.join(path, f"fold_{fold_num}.csv"))

# Dataset
train_dataset, test_dataset = dataset.get_kfold(fold_num, sample=False)

# Modeling
transfer_model = tf.keras.applications.efficientnet.EfficientNetB0(
    include_top=False, 
    weights=weight_option,
    input_shape=(224, 224, 3),
    pooling=None
)

loss_function = tf.keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=True,
    from_logits=True,
)

model = Model(
    transfer_model,
    loss_function
)
model = model.get_model()
model.summary()

# Visualize
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    verbose=1, # Show Progress Bar while Traning
    callbacks=[model_checkpoint_callback, csv_logger, early_stop_callback, reduce_lr_callback, lr_logging_callback]
)