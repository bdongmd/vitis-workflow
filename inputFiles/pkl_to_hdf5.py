import os
import pandas as pd
import numpy as np
import h5py

## Separate script for this process
## The pickle file was created with a version of python's `pickle` module that used protocol version 5
## But the python env used in Vitis AI container does not support this protocol version

with open('train_var_list.txt', 'r') as f1:
    lines = f1.readlines()

var_names = [line.strip() for line in lines]

pkl_data = pd.read_pickle('df_test.pkl')
subset_data = pkl_data[var_names]
labels = pkl_data['Sample']
#Y_test = np.eye(2,dtype=np.int32)[labels]

with h5py.File('subset_df_test.h5', 'w') as h5f:
    for col_name, col_data in subset_data.items():
        h5f.create_dataset(col_name, data=col_data)
    h5f.create_dataset('labels', data=labels)
    #h5f.create_dataset('Y_test', data=Y_test)
    h5f.close()

#outputfile = h5py.File('df_test.h5','w')
#outputfile.create_dataset('X_test', data=input_data, compression='gzip')
#outputfile.create_dataset('Y_test', data=Y_test,     compression='gzip')
#outputfile.create_dataset('labels', data=labels,     compression='gzip')
#outputfile.close()

