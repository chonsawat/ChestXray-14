""" Documations
a module for create a model easier for experimental
"""

import tensorflow as tf

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
        self.metrics = [tf.keras.metrics.AUC(multi_label=True)]
        self.transfer = transfer_model
        self.model = None
        
    def create_model(self, flatten=False):
        """ Create a Sequential of model for instances
        Example
        -------
        >>> self.create_model()
        """        
        sequential_list = [self.transfer]
        if flatten:
            sequential_list.append(tf.keras.layers.Flatten())
        sequential_list = sequential_list + [
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(15, activation='sigmoid')
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
        
    def get_model(self, flatten=False):
        """Create and Compile a Sequential of model for instances

        Returns
        -------
        tf.Model
            a Sequential of model
        """
        self.create_model(flatten)
        self.compile_model()
        return self.model
    
    def get_dropout_model(self, dropout_rate=0.5):
        sequential_list = [
            self.transfer,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            
            tf.keras.layers.Dense(15, activation='sigmoid')
        ]
        self.model = tf.keras.Sequential(sequential_list)
        self.compile_model()
        return self.model