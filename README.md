# vitis-workflow

This workflow has only been tested on hippeis (hippeis.pa.msu.edu).

## Setting up the workspace and dataset
1. Clone the repository by doing the following:
```
git clone git@github.com:bdongmd/vitis-workflow.git
```

2. Copy dataset
Since the dataset is not very small, I kept it on hippeis instead of uploading to github.\
If you don't have an hippeis account or have any issue access the file, please let me know.
```
cd vitis-workflow
cp /home/bdong/Xilinx/vitis-atlas/vitis-workflow/inputFiles/df_test.pkl inputFiles/
```

3. Start the Vitis AI docker and setup python virtual environment:
```
cd <path_to_directory>/vitis-workflow
singularity exec -H `pwd` -B /home/bdong,/ssd/home/bdong/Xilinx docker://xilinx/vitis-ai-cpu:latest bash
## you can also replace the cpu docker to the gpu one docker://xilinx/vitis-ai-gpu:latest
conda activate vitis-ai-tensorflow
```
Todo: make it compitable with python3.6

4. Run quantization
```
python main.py
```
