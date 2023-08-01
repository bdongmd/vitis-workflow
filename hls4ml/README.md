# hls4ml

This workflow has only been tested on hippeis (hippeis.pa.msu.edu).

hls4ml is added and tested here to make a comparison between the vitis AI and hls4ml.

2. Start the Vitis AI docker and setup python virtual environment:
```
cd hls4ml
## change the path after -B to your own path
singularity exec -H `pwd` -B /opt/Xilinx/Vivado/2019.2/,/home/bdong,/ssd/home/bdong/Xilinx docker://ghcr.io/fastmachinelearning/hls4ml-tutorial/hls4ml-0.7.1:latest bash
```
