import h5py
import numpy as np
from joblib import load

def load_data(inputFile="inputFiles/df_test.h5", scaleFile="inputFiles/std_scaler.bin", featureType=np.float32, labelType=np.int64):
	'''load testing dataset'''
	h5f = h5py.File(inputFile, 'r')
	scaler = load(scaleFile)
	features = np.array(h5f['X_test'], dtype=featureType)
	features = scaler.transform(features)
	labels = np.array(h5f['Y_test'], dtype=labelType)
	labels = np.argmax(labels, axis=-1)
	h5f.close()

	return features, labels
