import tensorflow as tf
from modules.models import Model

transfer_model = tf.keras.applications.resnet50.ResNet50(
    include_top=True, 
    weights=None,
    pooling='avg'
)
model = Model(transfer_model).get_model()
model.summary()