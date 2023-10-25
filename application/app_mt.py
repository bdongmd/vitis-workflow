import h5py
import numpy as np
import os
import argparse

def load_data_from_hdf5(hdf5_path):
	'''
	Load data from an HDF5 file.
	Input: Path to the HDF5 file
	Return: numpy array containing the data
	'''
	with h5py.File(hdf5_path, 'r') as f:
		data = np.array(f['data'])
		labels = np.array(f['labels'])
	return data, labels

def preprocess_data(data, scaler_file_path):
	'''
	Load a pre-trained scaler and use it to scale the input data.
	Input: data (numpy array), path to the scaler file
	Return: scaled data
	'''
	scaler = load(scaler_file_path)
	scaled_data = scaler.transform(data)
	quantized_data = data

def app(input_dir)

def main():

	ap = argparse.ArgumentParser()
	ap.add_argument('-d', '--input_dir', type=str, default='inputs', help='Path to folder of inputs. Default is inputs')
	ap.add_argument('-t', '--threads',   type=str, default=1,        help='Number of threads. Default is 1')
	ap.add_argument('-m', '--model',      type=str, default='customdnn.xmodel', help='Path of xmodel. Default is customdnn.xmodel')

	args = ap.parse_args()

	print('Command line options:')
	print(' --input_dir : ', args.image_dir)
	print(' --threads   : ', args.threads)
	print(' --model     : ', args.model)

	app(args.input_dire, args,threads, args.model)

if __name__ == '__main__':
	main()
