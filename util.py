import os
import json
import timeit
import h5py
import numpy as np
import pandas as pd
from joblib import load

def load_config(filename):
	with open(filename, "r") as f:
		return json.load(f)

def print_model(model):
	''' Print model structure '''
	print("------------------------------")
	for layer in model.layers:
		print(f"Layer name: {layer.name}")
		if layer.get_weights():
			weights = layer.get_weights()[0]
			print(f" - Weight Shape: {weights.shape}")
			print(f" - Weight Type: {weights.dtype}")
	print("------------------------------")

def load_dataset(dataset, scaler, verbose):
	with h5py.File(dataset, 'r') as h5f:
		labels = np.array(h5f['labels']) if 'labels' in h5f else None
		input_data = {name: np.array(data) for name, data in h5f.items() if name != 'labels'}
	if verbose:
		print("------------------------------")
		print("Data reading from: ", dataset)
		for col_name, data_array in input_data.items():
			print(f"-- Feature: {col_name}, Data Type: {data_array.dtype}")
	
	scaler = load(scaler)
	input_df = pd.DataFrame(input_data)
	scaled_array = scaler.transform(input_df)
	if verbose:
		print("------------------------------")
		print("After scaler:")
		scaled_data = {col_name: scaled_array[:, idx] for idx, col_name in enumerate(input_df.columns)}
		for col_name, data_array in scaled_data.items():
			print(f"-- Feature: {col_name}, Data Type: {data_array.dtype}")

	return labels, scaled_array

def time_prediction(model, inputdata, labels):
	start_time = timeit.default_timer()
	model_score = model.predict(inputdata)
	final_time = timeit.default_timer()
	latency = final_time - start_time
	throughput = len(labels) / latency
	return latency, throughput, model_score
