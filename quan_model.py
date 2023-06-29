import os
import argparse
import sys

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import pandas as pd
import h5py
import numpy as np

from tensorflow.keras.layers import Dropout
from tensorflow_model_optimization.quantization.keras import vitis_quantize

ap = argparse.ArgumentParser()
ap.add_argument('-m', '--float_model', type=str, default='inputFiles/best_model.h5', help='Path of floating-point model. Default is inputFiles/best_model.h5')
ap.add_argument('-q', '--quant_model', type=str, default='build/quant_model/best_q_model.h5', help='Path of quantized model. Default is build/quant_model/best_q_model.h5')
ap.add_argument('-b', '--batchsize',   type=int, default=50, help='Batchsize for quantization. Default is 50')
ap.add_argument('-e', '--evaluate',    action='store_true', help='Evaluate floating-point model if set. Default is no evaluation.')
args = ap.parse_args()

############### Print of info
print('\n------------------------------------')
print('TensorFlow version : ',tf.__version__)
print(sys.version)
print('------------------------------------')
print ('Command line options:')
print (' --float_model  : ', args.float_model)
print (' --quant_model  : ', args.quant_model)
print (' --batchsize    : ', args.batchsize)
print (' --evaluate     : ', args.evaluate)
print('------------------------------------\n')

def replace_dropout(loaded_model):
    new_model = tf.keras.models.clone_model(loaded_model)
    new_model.set_weights(loaded_model.get_weights())

    for layer in new_model.layers:
        if isinstance(layer, tf.keras.layers.Dropout):
            layer.build(layer.input_shape)
            layer.call = tf.function(layer.call)

    return new_model

def input_fn_quant(features, labels, batchsize):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.batch(batchsize, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

h5f = h5py.File('inputFiles/df_test.h5', 'r')
features = np.array(h5f['X_test'])
labels = np.array(h5f['Y_test'])
h5f.close()
labels = np.argmax(labels, axis=-1)

## load trained model
#model = tf.keras.models.load_model('./inputFiles/best_model.h5')
model = tf.keras.models.load_model('./inputFiles/my_model.h5')
print(model.summary())

## compile the original model
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

## Evaluate the orignal model
#print("\nEvaluating Original Model...")
#orig_loss, orig_acc, orig_auc = model.evaluate(features, labels)

## we want to quantize everything
## first here goes quantizing feature
features_quant = features.astype(np.float16)
## (not sure) since we are dealing with classificaiton problem where the labels are categorical and represented as integer class label, so I guess there is not need to quantize them?
# Create a tf.data.Dataset for calibration
calib_dataset = input_fn_quant(features_quant, labels, batchsize=50)

## Applying Quantization using Vitis Quantizer
quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(calib_dataset=calib_dataset, verbose=2)

## Compile and retrain the model
quantized_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
## Retrain the model after quantization (not sure if we want this so far, so commeting it out)
## quantized_model.fit(features_quant, labels, batch_size=32, epochs=5)

# Evaluate the Quantized Model with quantized features
print("\nEvaluating Quantized Model...")
quant_loss, quant_acc, quant_auc = quantized_model.evaluate(features_quant, labels)

# Compare original model performance with quantized model
print("\nOriginal Model Accuracy: {:.2f}, AUC: {:.2f}".format(orig_acc, orig_auc))
print("Quantized Model Accuracy: {:.2f}, AUC: {:.2f}".format(quant_acc, quant_auc))

