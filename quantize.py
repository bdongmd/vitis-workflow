import os
import util
import argparse
from prettytable import PrettyTable

import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def main():
	labels, inputdata = util.load_dataset(CONFIG['inputfile'], CONFIG['scaler'], CONFIG['verbose'])
	#quant_data = (inputdata * 255).astype(np.uint8)
	quant_data = inputdata
	model = load_model(CONFIG['float_model'])
	if CONFIG['verbose']:
		util.print_model(model)
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	# predict original model
	orig_lat, orig_thr, orig_pred = util.time_prediction(model, inputdata, labels)
	orig_size = os.path.getsize(CONFIG['float_model'])
	orig_pred_labels = (orig_pred > 0.5).astype(int).flatten()
	orig_acc = np.mean(orig_pred_labels == labels)

	# quantize model
	quantizer = vitis_quantize.VitisQuantizer(model)
	quantized_model = quantizer.quantize_model(calib_dataset=inputdata)
	quantized_model.save(CONFIG['quant_model'])
	if CONFIG['verbose']:
		util.print_model(quantized_model)
	quantized_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	quant_lat, quant_thr, quant_pred = util.time_prediction(quantized_model, quant_data, labels)
	quant_size = os.path.getsize(CONFIG['quant_model'])
	quant_pred_labels = (quant_pred > 0.5).astype(int).flatten()
	quant_acc = np.mean(quant_pred_labels == labels)

	# re-train quantized model if needed
	if CONFIG['retrain_model']:
		quant_data = inputdata # TODO: update this
		retrain_quantized_model.fit(quant_data, labels, batch_size=args.batchsize, epochs=5)
		retrain_quantized_model.save(CONFIG['retrain_model_path'])
		requant_lat, requant_thr, requant_size, requant_pred = util.time_prediction(retrain_quantized_model, quant_data, labels)
	
	# Compare model performance at different stage
	print("\nSummarizing the performance between the orignal and quantized models:")
	table = PrettyTable()
	table.field_names = ["Factor", "Original Model", "Quantized Model"]
	table.add_row(["Accuracy[%]", format(orig_acc*100, '.2f'), format(quant_acc*100, '.2f')])
	table.add_row(["Size (KB)", format(orig_size, '.2f'), format(quant_size, '.2f')])
	table.add_row(["Latency (s)", format(orig_lat, '.2f'), format(quant_lat, '.2f')])
	table.add_row(["Throughput", format(orig_thr, '.2f'), format(quant_thr,'.2f')])
	print(table)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Script to process data.')
	parser.add_argument('-c', '--config', type=str, default='config.json', help='Path to the configuration file')
	args = parser.parse_args()

	CONFIG = util.load_config(args.config)

	main()
