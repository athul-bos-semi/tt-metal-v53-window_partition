
#/bin/bash
# set -eo pipefail

run_t3000_ttmetal_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttmetal_tests"

  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsDirectSendAllConnectedChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsSendInterleavedBufferAllConnectedChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsDirectRingGatherAllChips" ; fail+=$?
  TT_METAL_SLOW_DISPATCH_MODE=1 ./build/test/tt_metal/unit_tests --gtest_filter="DeviceFixture.EthKernelsInterleavedRingGatherAllChips" ; fail+=$?
  TT_METAL_ENABLE_REMOTE_CHIP=1 ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter="CommandQueueSingleCardFixture.*" ; fail+=$?
  ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter="CommandQueueMultiDeviceFixture.*" ; fail+=$?
  ./build/test/tt_metal/unit_tests_fast_dispatch --gtest_filter="DPrintFixture.*:WatcherFixture.*" ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttmetal_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_ttnn_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_ttnn_tests"
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml ./build/test/ttnn/unit_tests_ttnn
  pytest -n auto tests/ttnn/unit_tests/test_multi_device_trace.py ; fail+=$?
  WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest tests/ttnn/unit_tests/test_multi_device_trace.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/test_multi_device.py ; fail+=$?
  pytest -n auto tests/ttnn/unit_tests/test_multi_device_async.py ; fail+=$?
  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_ttnn_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_falcon7b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon7b_tests"

  pytest -n auto models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_mlp.py ; fail+=$?
  pytest -n auto models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_attention.py ; fail+=$?
  pytest -n auto models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_decoder.py ; fail+=$?
  #pytest models/demos/ttnn_falcon7b/tests/multi_chip/test_falcon_causallm.py

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon7b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_falcon40b_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_falcon40b_tests"

  # Failing - Issue #10321
  # WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest -n auto models/demos/t3000/falcon40b/tests/ci/test_falcon_end_to_end_1_layer_t3000.py ; fail+=$?


  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_falcon40b_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_mixtral_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_mixtral_tests"

  # Failing - Issue #10322
  #pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_attention.py ; fail+=$?
  #pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_mlp.py ; fail+=$?
  #pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_rms_norm.py ; fail+=$?
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_embedding.py ; fail+=$?
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_moe.py ; fail+=$?
  pytest -n auto models/demos/t3000/mixtral8x7b/tests/test_mixtral_decoder.py ; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_mixtral_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tests() {
  # Run ttmetal tests
  run_t3000_ttmetal_tests

  # Run ttnn tests
  run_t3000_ttnn_tests

  # Run falcon7b tests
  run_t3000_falcon7b_tests

  # Run falcon40b tests
  run_t3000_falcon40b_tests

  # Run mixtral tests
  run_t3000_mixtral_tests
}

fail=0
main() {
  # For CI pipeline - source func commands but don't execute tests if not invoked directly
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_t3000_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
