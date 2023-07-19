# vitis-workflow

This workflow has only been tested on hippeis (hippeis.pa.msu.edu).

## Setting up the workspace and dataset
1. Clone the repository by doing the following:
```
git clone git@github.com:bdongmd/vitis-workflow.git
```

2. Copy dataset:

Since the dataset is not so small, I am keeping it on hippeis instead of uploading to github.\
If you don't have an hippeis account or have any issue access the file, please let me know.
```
cd vitis-workflow
cp /home/bdong/Xilinx/vitis-atlas/vitis-workflow/inputFiles/df_test.h5 inputFiles/
```

3. Start the Vitis AI docker and setup python virtual environment:
```
cd vitis-workflow
## change the path after -B to your own path
singularity exec -H `pwd` -B /home/bdong,/ssd/home/bdong/Xilinx docker://xilinx/vitis-ai-cpu:latest bash
## you can also replace the cpu docker to the gpu one docker://xilinx/vitis-ai-gpu:latest
conda activate vitis-ai-tensorflow2
pip install prettytable --user
```

4. Run quantization
```
python quan_model.py -m inputFiles/my_model.h5
```

You'll get a table summarizeing the facotrs you are comparing between the original and quantized model as following:
```
Summarizing the performance between the orignal and quantized models:
+-------------+--------------------+--------------------+
|    Factor   |   Original Model   |  Quantized Model   |
+-------------+--------------------+--------------------+
|   Accuracy  |        0.34        |        0.34        |
|     AUC     |        0.43        |        0.44        |
|  Size (KB)  |       85.93        |       67.93        |
| Latency (s) | 13.652668051421642 | 18.273577198386192 |
|  Throughput |      33841.08      |      25283.56      |
+-------------+--------------------+--------------------+
```
The quantized model is saved under `../output/quantized_model.h5`
