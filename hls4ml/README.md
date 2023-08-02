# hls4ml

This workflow has only been tested on hippeis (hippeis.pa.msu.edu).

hls4ml is added and tested here to make a comparison between the vitis AI and hls4ml.

0. download required python libraries:
Most of the libraries needed already exist on hippeis, only the following needs to be donwload
```
pip install --user hls4ml
pip install --user tensorflow==2.6.2
```

1. Setup Vivado 
```
alias setup_vivado_2019p2="source /opt/Xilinx/Vivado/2019.2/settings64.sh"
setup_vivado_2019p2
```
