import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from prep_data import *
 
def create_model():
    # en entrée on met 768 features (12 pièces x 8 rangées x 8 colonnes)
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(768,)),  # Couche 1 + couche d'entrée cachée
        layers.Dense(256, activation='relu'),                     # Couche 2
        layers.Dense(1)                                           # Couche de sortie
    ])
    return model