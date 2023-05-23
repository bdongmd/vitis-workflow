# vitis-workflow

This workflow has only been tested on hippeis (hippeis.pa.msu.edu).

##Setting up the workspace and dataset
1. Clone the repository by doing the following:
```
git clone git@github.com:bdongmd/vitis-workflow.git
```

2. Start the Vitis AI docker:
```
cd <path_to_directory>/vitis-workflow
singularity run docker://xilinx/vitis-ai-cpu:latest ## for cpu
singularity run docker://xilinx/vitis-ai-gpu:latest ## for gpu
```

3. 
