# hls4ml

This workflow has only been tested on hippeis (hippeis.pa.msu.edu).

hls4ml is added and tested here to make a comparison between the vitis AI and hls4ml.

## Setting up environment via conda:

Note: the package included in this environment has only been tested for the part1 example. For the other examples [here](https://github.com/fastmachinelearning/hls4ml-tutorial/tree/main) may need more packages to be included in the `pip`.
```
conda env create -f environment.yml
conda activate hls4ml-tutorial
alias setup_vivado_2019p2="source /opt/Xilinx/Vivado/2019.2/settings64.sh"
setup_vivado_2019p2
python3 part1_getting_started.py
```

