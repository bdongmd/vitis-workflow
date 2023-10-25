# Vitis AI model quantization flow

This is used to quantize the NN models using the TensorFlow2 based on the Vitis AI ([documentation protal](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Overview?tocId=froN06fqkXdB2HpfEBtHNw)) .

This workflow has only been tested on hippeis (hippeis.pa.msu.edu).

The Vitis AI quantizer accepts a floating-point model as input and performs pre-processing (folds batchnorms and removes nodes not required for interference). It then quantizes the weihgts/biases and activations to the given bit width.

## Structure
* [Quantization steps](#quantization-steps)
* [Instructions](#instructions)

## Quantization steps
- Inspect the float model
  `VitisInspector` is a helper tool that inspects a float model, shows partition results for a given DPU target architecture, and indicates why the layeres are not mapped to DPU. 
- Use quantizer to quantize the model
  Two approaches are avaiable: post-training quantization and quantization aware training
  post-trianing quantization: float model and calibration set should be avaiable for this step.
  quantization aware training: 
- Evaluate the quantized model
  Same way to evaluate the float model. (Q: would the input need to be quantized?)
-  

## Instructions
1. Clone the repository by doing the following:
```
git clone git@github.com:bdongmd/vitis-workflow.git
```

2. Start the Vitis AI docker and setup python virtual environment:
```
cd vitis-workflow
## change the path after -B to your own path
singularity exec -H `pwd` -B /home/bdong,/ssd/home/bdong/Xilinx docker://xilinx/vitis-ai-cpu:latest bash
## you can also replace the cpu docker to the gpu one docker://xilinx/vitis-ai-gpu:latest
conda activate vitis-ai-tensorflow2
pip install prettytable --user
```

3. Run quantization
```
python quan_model.py -m inputFiles/best_model.h5
```

You'll get a table summarizeing the facotrs you are comparing between the original and quantized model as following:
```
Summarizing the performance between the orignal and quantized models:
+-------------+----------------+-----------------+
|    Factor   | Original Model | Quantized Model |
+-------------+----------------+-----------------+
|   Accuracy  |      0.66      |       0.66      |
|     AUC     |      0.66      |       0.66      |
|  Size (KB)  |     85.93      |      67.93      |
| Latency (s) |     13.55      |      19.73      |
|  Throughput |    34087.07    |     23421.07    |
+-------------+----------------+-----------------+
```
The quantized model is saved under `../output/quantized_model.h5`
