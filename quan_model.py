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
print('------------------------------------\n')

h5f = h5py.File('inputFiles/df_test.h5', 'r')
features = np.array(h5f['X_test'], dtype=np.float32)
labels = np.array(h5f['Y_test'], dtype=np.int64)
labels = np.argmax(labels, axis=-1)
h5f.close()

## load trained model
model = tf.keras.models.load_model(args.float_model)
print(model.summary())

## compile the original model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

## Evaluate the orignal model
print("\nEvaluating Original Model...")
orig_loss, orig_acc, orig_auc = model.evaluate(features, labels, 32)

## we want to quantize everything
## first here goes quantizing feature
features_quant = features.astype(np.float16)

## Applying Quantization using Vitis Quantizer
quantizer = vitis_quantize.VitisQuantizer(model)
quantized_model = quantizer.quantize_model(calib_dataset=features_quant)

## Compile and retrain the model
quantized_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
## Retrain the model after quantization (not sure if we want this so far, so commeting it out)
## quantized_model.fit(features_quant, labels, batch_size=args.batchsize, epochs=5)

# Evaluate the Quantized Model with quantized features
print("\nEvaluating Quantized Model...")
quant_loss, quant_acc, quant_auc = quantized_model.evaluate(features_quant, labels)

# Compare original model performance with quantized model
print("\nOriginal Model Accuracy: {:.2f}, AUC: {:.2f}".format(orig_acc, orig_auc))
print("Quantized Model Accuracy: {:.2f}, AUC: {:.2f}".format(quant_acc, quant_auc))

