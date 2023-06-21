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

with open('inputFiles/train_var_list.txt', 'r') as f1:
    lines = f1.readlines()

var_names = [line.strip() for line in lines]

pkl_data = pd.read_pickle('inputFiles/df_test.pkl')
input_data = pkl_data[var_names]
labels = pkl_data['Sample']

### To check the range of input, here assumed quatization to 8-bit unsigned intergers (tf.quint8) with a range of 0 - 255
### I may need to normalize input_data...
quantized_data =  tf.quantization.quantize(input_data, 0, 255, tf.quint8)
dataset = tf.data.Dataset.from_tensor_slices((input_data, labels))

quant_model(args.float_model, args.quant_model, args.batchsize, dataset, args.evaluate)
