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
        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.metrics = [tf.keras.metrics.AUC(multi_label=True),
                        tf.keras.metrics.Precision(thresholds=0),
                        tf.keras.metrics.Recall(thresholds=0),
                        tfa.metrics.F1Score(num_classes=15,
                                            threshold=0.5)]
        self.transfer = transfer_model
        self.model = None
        
    def create_model(self):
        """ Create a Sequential of model for instances
        Example
        -------
        >>> self.create_model()
        """        
        self.model = tf.keras.Sequential([
            self.transfer,
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(15, activation='sigmoid')]
        )
    
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
    