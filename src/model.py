import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from prep_data import *
 
def create_model():
    # en entrée on met 768 features (12 pièces x 8 rangées x 8 colonnes)
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(768,)),  # Couche 1 + couche d'entrée cachée
        # layers.PReLU(),  
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # Couche 2
        layers.Dropout(0.3),
        # layers.PReLU(),  
        # layers.Dense(128, activation='relu'),     # Couche 3
        # layers.Dense(64,activation='relu'),      # Couche 4
        # layers.PReLU(),  
        layers.Dense(1)        # Couche de sortie
    ])
    return model