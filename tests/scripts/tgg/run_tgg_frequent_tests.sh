
#/bin/bash
set -eo pipefail

run_tgg_tests() {
  # Add tests here
  echo "LOG_METAL: running run_tgg_frequent_tests"
  pytest -n auto tests/ttnn/unit_tests/test_multi_device_trace_tgg.py --timeout=1500 ; fail+=$?
}

main() {
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

  run_tgg_tests
}

main "$@"
