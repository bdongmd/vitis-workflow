'''
Make target folder
Copy images, application code and compiled model to 'target'
'''

import argparse
import os
import shutil
import sys

DIVIDER = '-----------------------------------------'

def make_target(target_dir, input_dir, app_dir, model):

	# remove any previous data
	shutil.rmtree(target_dir, ignore_errors=True)    
	os.makedirs(target_dir)
	os.makedirs(target_dir+'/inputs')

	# save inputs to target folder
	print('Copying input dataset from ', input_dir, '...')
	shutil.copy(os.path.join(input_dir, 'test.h5'), target_dir)

	# copy application code
	print('Copying application code from ', app_dir, '...')
	shutil.copy(os.path.join(app_dir, 'app_mt.py'), target_dir)

	# copy compiled model
	print('Copying compiled model from ', model, '...')
	shutil.copy(model, target_dir)

	return

def main():

	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument('-t', '--target_dir', type=str, default='build/target', help='Full path of target folder. Default is build/target')
	ap.add_argument('-i', '--input_dir',  type=str, default='build/dataset/test', help='Full path of images folder. Default is build/dataset/test')
	ap.add_argument('-a', '--app_dir',    type=str, default='application', help='Full path of application code folder. Default is application')
	ap.add_argument('-m', '--model',      type=str,  default='build/compiled_model/customdnn.xmodel', help='Full path of compiled model.Default is build/compiled_model/customdnn.xmodel')
	args = ap.parse_args()

	make_target(args.target_dir, args.input_dir, args.app_dir, args.model)

if __name__ == "__main__":
	main()
