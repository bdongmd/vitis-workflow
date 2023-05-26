import os

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import pandas ad pd

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

