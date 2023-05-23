import argparse
import os
import shutil
import sys

from quantize import quant_model

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import pandas as pd
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

with open('inputFiles/train_var_list.txt', 'r') as f1:
    lines = f1.readlines()

var_names = [line.strip() for line in lines]

pkl_data = pd.read_pickle('inputFiles/df_test.pkl')
input_data = pkl_data[var_names]
labels = pkl_data['Sample']


