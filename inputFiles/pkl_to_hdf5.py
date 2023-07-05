import os
import pandas as pd
import numpy as np
import h5py

with open('train_var_list.txt', 'r') as f1:
    lines = f1.readlines()

var_names = [line.strip() for line in lines]

pkl_data = pd.read_pickle('df_test.pkl')
input_data = pkl_data[var_names]
labels = pkl_data['Sample']
Y_test = np.eye(2,dtype=int)[labels]

outputfile = h5py.File('df_test.h5','w')
outputfile.create_dataset('X_test', data=input_data, compression='gzip')
outputfile.create_dataset('Y_test', data=Y_test,     compression='gzip')
outputfile.create_dataset('labels', data=labels,     compression='gzip')
outputfile.close()

