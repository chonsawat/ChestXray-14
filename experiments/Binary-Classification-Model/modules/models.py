import tensorflow as tf

class ModelBinary:
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
        
    def create_model(self):
        """ Create a Sequential of model for instances
        Example
        -------
        >>> self.create_model()
        """
        sequential_list = [
            self.transfer,
            tf.keras.layers.Flatten(),
            
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
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

    
class ModelBinaryWithDropout:
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
        
    def create_model(self):
        """ Create a Sequential of model for instances
        Example
        -------
        >>> self.create_model()
        """
        sequential_list = [
            self.transfer,
            tf.keras.layers.Flatten(),
            
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
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