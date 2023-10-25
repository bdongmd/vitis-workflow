import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Input, Dropout
from tensorflow.keras.models import Model
import h5py
import numpy as np

def build_model(input_shape):
    In = Input(shape=[input_shape,])
    x = In
    x = Dense(30)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs = In, outputs=x)

    model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
    )

    return model

h5f = h5py.File('df_test.h5', 'r')
features = np.array(h5f['X_test'])
labels = np.array(h5f['Y_test'])
h5f.close()
labels = np.argmax(labels, axis=-1)

# Get the shape of the input features
input_shape = features.shape[1]

# Create a sequential model
model = build_model(features.shape[1]) 
# Train the model
model.fit(features, labels, epochs=1, batch_size=32, validation_split=0.2)

model.save('my_model.h5')
