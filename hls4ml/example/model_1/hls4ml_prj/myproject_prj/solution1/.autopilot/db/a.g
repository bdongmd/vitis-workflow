#!/bin/sh
lli=${LLVMINTERP-lli}
exec $lli \
    /ssd/home/bdong/Xilinx/vitis-atlas/vitis-workflow/hls4ml/example/model_1/hls4ml_prj/myproject_prj/solution1/.autopilot/db/a.g.bc ${1+"$@"}
