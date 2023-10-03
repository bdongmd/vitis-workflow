#!/bin/sh

conda activate vitis-ai-tensorflow2

# folders
export BUILD=./build
export LOG=${BUILD}/logs
mkdir -p ${LOG}

# list of GPUs to use - modify as required for your system
# TODO might need to revist this
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="0"

# training
# add training here if we want to do it later on

# quantize & evaluate
python quantize.py


# compile for selected target board
source compile.sh zcu102
source compile.sh zcu104
source compile.sh vck190
source compile.sh u50

# make target folder

