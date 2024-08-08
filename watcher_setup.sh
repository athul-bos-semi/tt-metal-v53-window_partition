rm -rf *.log

#tt-smi-metal -r 0,1,2,3

unset TT_METAL_WATCHER

unset TT_METAL_WATCHER_DISABLE_ASSERT
unset TT_METAL_WATCHER_DISABLE_PAUSE
unset TT_METAL_WATCHER_DISABLE_RING_BUFFER
unset TT_METAL_WATCHER_DISABLE_STATUS
unset TT_METAL_WATCHER_NOINLINE

unset TT_METAL_WATCHER_DISABLE_NOC_SANITIZE

unset TT_METAL_WATCHER_DEBUG_DELAY
unset TT_METAL_READ_DEBUG_DELAY_CORES
unset TT_METAL_WRITE_DEBUG_DELAY_CORES
unset TT_METAL_READ_DEBUG_DELAY_RISCVS
unset TT_METAL_WRITE_DEBUG_DELAY_RISCVS

#export TT_METAL_WATCHER=1

#export TT_METAL_WATCHER_DISABLE_ASSERT=1
#export TT_METAL_WATCHER_DISABLE_PAUSE=1
#export TT_METAL_WATCHER_DISABLE_RING_BUFFER=1
##export TT_METAL_WATCHER_DISABLE_STATUS=1
#export TT_METAL_WATCHER_NOINLINE=1

#export TT_METAL_WATCHER_DISABLE_NOC_SANITIZE=1

#export TT_METAL_WATCHER_DEBUG_DELAY=10
#export TT_METAL_READ_DEBUG_DELAY_CORES=0,0
#export TT_METAL_WRITE_DEBUG_DELAY_CORES=0,0
#export TT_METAL_READ_DEBUG_DELAY_RISCVS=BR
#export TT_METAL_WRITE_DEBUG_DELAY_RISCVS=BR

unset TT_METAL_CLEAR_L1
#export TT_METAL_CLEAR_L1=1

unset WH_ARCH_YAML
#export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml

unset TT_METAL_DEVICE_PROFILER
#export TT_METAL_DEVICE_PROFILER=1

unset TT_METAL_DPRINT_CORES
export TT_METAL_DPRINT_CORES=4,7

unset TT_METAL_DPRINT_FILE
export TT_METAL_DPRINT_FILE=dprint_1.log
pytest test.py
mv ./before.log ./before_1.log
mv ./after.log ./after_1.log
mv ./close.log ./close_1.log

#unset TT_METAL_DPRINT_FILE
#export TT_METAL_DPRINT_FILE=dprint_2.log
#pytest test.py
#mv ./before.log ./before_2.log
#mv ./after.log ./after_2.log
#mv ./close.log ./close_2.log

cat -n ./dprint_1.log
cat -n ./after_1.log
cat -n ./close_1.log
diff ./after_1.log ./close_1.log
