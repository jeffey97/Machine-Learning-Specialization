import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("Devices:", tf.config.list_physical_devices())

# Tiny dummy model
model = keras.Sequential([
    layers.Dense(32, activation="relu", input_shape=(100,)),
    layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")

import numpy as np
x = np.random.randn(1000, 100)  # 1000 samples, 100 features
y = np.random.randn(1000, 1)

model.fit(x, y, epochs=3, batch_size=32)
