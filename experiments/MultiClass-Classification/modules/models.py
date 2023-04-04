""" Documations
a module for create a model easier for experimental
"""

import tensorflow as tf
import tensorflow_addons as tfa

class Model:
    """ Use for create a different transfer learning model
    Example
    --------
    >>> from modules.models import Model
    >>> transfer_model = tf.keras.applications.resnet50.ResNet50(
        include_top=True, 
        weights=None,
        pooling='avg'
    )
    >>> model = Model(transfer_model).get_model()
    """
    def __init__(self, transfer_model):
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.metrics = [tfa.metrics.F1Score(num_classes=4), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        self.transfer = transfer_model
        self.model = None
        
    def create_model(self):
        """ Create a Sequential of model for instances
        Example
        -------
        >>> self.create_model()
        """        
        sequential_list = [
            self.transfer,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(4, activation='softmax')
        ]
        self.model = tf.keras.Sequential(sequential_list)
    
    def compile_model(self):
        """Compile a Sequential of model for instances
        Example
        -------
        >>> self.compile_model()
        """        
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)
        
    def get_model(self):
        """Create and Compile a Sequential of model for instances

        Returns
        -------
        tf.Model
            a Sequential of model
        """
        self.create_model()
        self.compile_model()
        return self.model
    