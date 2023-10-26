# Vitis AI model quantization flow

This is used to quantize the NN models using the TensorFlow2 based on the Vitis AI ([documentation protal](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Overview?tocId=froN06fqkXdB2HpfEBtHNw)) .

This workflow has only been tested on hippeis (hippeis.pa.msu.edu).

The Vitis AI quantizer accepts a floating-point model as input and performs pre-processing (folds batchnorms and removes nodes not required for interference). It then quantizes the weihgts/biases and activations to the given bit width.

## Structure
* [Quantization steps](#quantization-steps)
* [Instructions](#instructions)
* [References](#references)

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
python quantize.py -c config/maria_model.json
```

## References
The code is developed based on the examples from [Vitis-AI tutorial](https://github.com/Xilinx/Vitis-AI-Tutorials/), especially, [02-MNIST_classification_tf](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/1.4/Design_Tutorials/02-MNIST_classification_tf) and [08-tf2_flow](https://github.com/Xilinx/Vitis-AI-Tutorials/tree/1.4/Design_Tutorials/08-tf2_flow).
