# hls4ml

This workflow has only been tested on hippeis (hippeis.pa.msu.edu).

hls4ml is added and tested here to make a comparison between the vitis AI and hls4ml.

## Setting up environment via conda:

Note: the package included in this environment has only been tested for the part1 example. For the other examples [here](https://github.com/fastmachinelearning/hls4ml-tutorial/tree/main) may need more packages to be included in the `pip`.
To create the `hls4ml-tutorial` env the first time you use this by doing:
```
conda env create -f environment.yml
```

Once the env is created, later on you can activate and deactivate the env by doing:
```
conda activate hls4ml-tutorial ## activate the env
conda deactivate ## deactivate the env
```
In case you want to remove the env:
```
conda env remove --name hls4ml-tutorial
```

To set up Vivada environment, once you activated the `hls4ml-tutorial`:
```
alias setup_vivado_2019p2="source /opt/Xilinx/Vivado/2019.2/settings64.sh"
setup_vivado_2019p2
```
## Run Example
To run example:
```
cd example
python3 hls4ml_part1_example.py
```
