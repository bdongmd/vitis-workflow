import os
import argparse

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import pandas as pd

from quantize import quant_model

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

features = pd.read_hdf('inputFiles/df_test.h5', key='X_test')
labels   = pd.read_hdf('inputFiles/df_test.h5', key='Y_test')

# Convert features and labels to Tensor
features_tensor = tf.convert_to_tensor(features.values, dtype=tf.float32)
labels_tensor = tf.convert_to_tensor(labels.values, dtype=tf.float32)

# Quantize the features using tf.quantization
# Quantize the features to tf.qint8, you might need to adjust this later on
features_min, features_max = tf.reduce_min(features_tensor), tf.reduce_max(features_tensor)
_, features_scale, features_offset = tf.quantization.quantize(features_tensor, features_min, features_max, tf.qint8)

# Now, you have quantized features that you can use for further processing
quantized_features = tf.quantization.quantize_and_dequantize(features_tensor, features_min, features_max)


quant_model(args.float_model, args.quant_model, args.batchsize, dataset, args.evaluate)
