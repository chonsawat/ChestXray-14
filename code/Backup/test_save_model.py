import os
import wandb
import tensorflow as tf
from wandb.keras import WandbCallback
from modules.models import Model

# Dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# WandbCallback
run = wandb.init(project="Expepriment Temporary", name="Saving model 6")

# Modeling
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.6),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[WandbCallback()])

print(model.evaluate(test_images, test_labels))
wandb.save("model.h5")

# model = tf.keras.models.load_model(r"E:\Python\ChestXray-14\wandb\run-20220907_160322-3c0klkx1\files\model-best.h5")
# print(model.evaluate(test_images, test_labels))