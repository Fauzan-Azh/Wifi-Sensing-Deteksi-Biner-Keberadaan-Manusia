# File: model_terdahulu.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_previous_model(input_shape):
    # Arsitektur ini mereplikasi LSTM + Attention dari Tabel 4.22 Laporan TA
    inputs = layers.Input(shape=input_shape)
    
    # Layer LSTM sesuai laporan
    x = layers.LSTM(64, return_sequences=True)(inputs) 
    
    # Layer Attention
    x = layers.Attention()([x, x])
    
    # Flattening
    x = layers.Flatten()(x)
    
    # Dense Layer
    x = layers.Dense(16, activation='relu')(x)
    
    # Output Layer (Biner)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="Previous_Research_Model")