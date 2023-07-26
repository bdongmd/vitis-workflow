# vitis-workflow

This workflow has only been tested on hippeis (hippeis.pa.msu.edu).

## Setting up the workspace and dataset
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
