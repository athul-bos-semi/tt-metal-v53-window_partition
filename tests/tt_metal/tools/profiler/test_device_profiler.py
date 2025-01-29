# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os, sys
import json
import re
import inspect
import pytest
import subprocess

import pandas as pd
import numpy as np

from tt_metal.tools.profiler.common import (
    TT_METAL_HOME,
    PROFILER_HOST_DEVICE_SYNC_INFO,
    PROFILER_SCRIPTS_ROOT,
    PROFILER_ARTIFACTS_DIR,
    PROFILER_LOGS_DIR,
    clear_profiler_runtime_artifacts,
)

from models.utility_functions import skip_for_grayskull

PROG_EXMP_DIR = "programming_examples/profiler"


def get_device_data(setupStr=""):
    postProcessRun = os.system(
        f"cd {PROFILER_SCRIPTS_ROOT} && " f"./process_device_log.py {setupStr} --no-artifacts --no-print-stats"
    )

    assert postProcessRun == 0, f"Log process script crashed with exit code {postProcessRun}"

    devicesData = {}
    with open(f"{PROFILER_ARTIFACTS_DIR}/output/device/device_analysis_data.json", "r") as devicesDataJson:
        devicesData = json.load(devicesDataJson)

    return devicesData


def run_gtest_profiler_test(testbin, testname):
    clear_profiler_runtime_artifacts()
    output = subprocess.check_output(
        f"cd {TT_METAL_HOME} && {testbin} --gtest_filter={testname}", stderr=subprocess.STDOUT, shell=True
    ).decode("UTF-8")
    print(output)
    if "SKIPPED" not in output:
        get_device_data()


def run_device_profiler_test(testName=None, setupAutoExtract=False, slowDispatch=False):
    name = inspect.stack()[1].function
    testCommand = f"build/{PROG_EXMP_DIR}/{name}"
    if testName:
        testCommand = testName
    print("Running: " + testCommand)
    clear_profiler_runtime_artifacts()
    slowDispatchEnv = ""
    if slowDispatch:
        slowDispatchEnv = "TT_METAL_SLOW_DISPATCH_MODE=1 "
    profilerRun = os.system(f"cd {TT_METAL_HOME} && {slowDispatchEnv}{testCommand}")
    assert profilerRun == 0

    setupStr = ""
    if setupAutoExtract:
        setupStr = f"-s {name}"

    return get_device_data(setupStr)


def get_function_name():
    frame = inspect.currentframe()
    return frame.f_code.co_name


@skip_for_grayskull()
def test_multi_op():
    OP_COUNT = 1000
    RUN_COUNT = 2
    REF_COUNT_DICT = {
        "grayskull": [108 * OP_COUNT * RUN_COUNT, 88 * OP_COUNT * RUN_COUNT],
        "wormhole_b0": [72 * OP_COUNT * RUN_COUNT, 64 * OP_COUNT * RUN_COUNT, 56 * OP_COUNT * RUN_COUNT],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setupAutoExtract=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    statName = f"BRISC KERNEL_START->KERNEL_END"

    assert statName in stats.keys(), "Wrong device analysis format"
    assert stats[statName]["stats"]["Count"] in REF_COUNT_DICT[ENV_VAR_ARCH_NAME], "Wrong Marker Repeat count"


def test_custom_cycle_count_slow_dispatch():
    REF_CYCLE_COUNT_PER_LOOP = 52
    LOOP_COUNT = 2000
    REF_CYCLE_COUNT = REF_CYCLE_COUNT_PER_LOOP * LOOP_COUNT
    REF_CYCLE_COUNT_HIGH_MULTIPLIER = 10
    REF_CYCLE_COUNT_LOW_MULTIPLIER = 5

    REF_CYCLE_COUNT_MAX = REF_CYCLE_COUNT * REF_CYCLE_COUNT_HIGH_MULTIPLIER
    REF_CYCLE_COUNT_MIN = REF_CYCLE_COUNT // REF_CYCLE_COUNT_LOW_MULTIPLIER

    devicesData = run_device_profiler_test(setupAutoExtract=True, slowDispatch=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    for risc in ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]:
        statName = f"{risc} KERNEL_START->KERNEL_END"

        assert statName in stats.keys(), "Wrong device analysis format"
        assert stats[statName]["stats"]["Average"] < REF_CYCLE_COUNT_MAX, "Wrong cycle count, too high"
        assert stats[statName]["stats"]["Average"] > REF_CYCLE_COUNT_MIN, "Wrong cycle count, too low"


def test_custom_cycle_count():
    REF_CYCLE_COUNT_PER_LOOP = 52
    LOOP_COUNT = 2000
    REF_CYCLE_COUNT = REF_CYCLE_COUNT_PER_LOOP * LOOP_COUNT
    REF_CYCLE_COUNT_HIGH_MULTIPLIER = 10
    REF_CYCLE_COUNT_LOW_MULTIPLIER = 5

    REF_CYCLE_COUNT_MAX = REF_CYCLE_COUNT * REF_CYCLE_COUNT_HIGH_MULTIPLIER
    REF_CYCLE_COUNT_MIN = REF_CYCLE_COUNT // REF_CYCLE_COUNT_LOW_MULTIPLIER

    devicesData = run_device_profiler_test(setupAutoExtract=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    for risc in ["BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"]:
        statName = f"{risc} KERNEL_START->KERNEL_END"

        assert statName in stats.keys(), "Wrong device analysis format"
        assert stats[statName]["stats"]["Average"] < REF_CYCLE_COUNT_MAX, "Wrong cycle count, too high"
        assert stats[statName]["stats"]["Average"] > REF_CYCLE_COUNT_MIN, "Wrong cycle count, too low"


def test_full_buffer():
    OP_COUNT = 26
    RISC_COUNT = 5
    ZONE_COUNT = 125
    REF_COUNT_DICT = {
        "grayskull": [108 * OP_COUNT * RISC_COUNT * ZONE_COUNT, 88 * OP_COUNT * RISC_COUNT * ZONE_COUNT],
        "wormhole_b0": [
            72 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
            64 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
            56 * OP_COUNT * RISC_COUNT * ZONE_COUNT,
        ],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setupAutoExtract=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]
    statName = "Marker Repeat"
    statNameEth = "Marker Repeat ETH"

    assert statName in stats.keys(), "Wrong device analysis format"

    if statNameEth in stats.keys():
        assert (
            stats[statName]["stats"]["Count"] - stats[statNameEth]["stats"]["Count"]
            in REF_COUNT_DICT[ENV_VAR_ARCH_NAME]
        ), "Wrong Marker Repeat count"
        assert stats[statNameEth]["stats"]["Count"] > 0, "Wrong Eth Marker Repeat count"
        assert stats[statNameEth]["stats"]["Count"] % (OP_COUNT * ZONE_COUNT) == 0, "Wrong Eth Marker Repeat count"
    else:
        assert stats[statName]["stats"]["Count"] in REF_COUNT_DICT[ENV_VAR_ARCH_NAME], "Wrong Marker Repeat count"


def test_dispatch_cores():
    OP_COUNT = 1
    RISC_COUNT = 1
    ZONE_COUNT = 37
    REF_COUNT_DICT = {
        "grayskull": {
            "Tensix CQ Dispatch": 16,
            "Tensix CQ Prefetch": 25,
        },
        "wormhole_b0": {
            "Tensix CQ Dispatch": 16,
            "Tensix CQ Prefetch": 25,
        },
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "1"

    devicesData = run_device_profiler_test(setupAutoExtract=True)

    stats = devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["analysis"]

    verifiedStat = []
    for stat in REF_COUNT_DICT[ENV_VAR_ARCH_NAME].keys():
        if stat in stats.keys():
            verifiedStat.append(stat)
            assert stats[stat]["stats"]["Count"] == REF_COUNT_DICT[ENV_VAR_ARCH_NAME][stat], "Wrong Dispatch zone count"

    statTypes = ["Dispatch", "Prefetch"]
    statTypesSet = set(statTypes)
    for statType in statTypes:
        for stat in verifiedStat:
            if statType in stat:
                statTypesSet.remove(statType)
    assert len(statTypesSet) == 0
    os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"


@skip_for_grayskull()
def test_ethernet_dispatch_cores():
    REF_COUNT_DICT = {
        "Ethernet CQ Dispatch": [17, 12, 3902],
        "Ethernet CQ Prefetch": [18, 1954],
    }
    os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "1"
    devicesData = run_device_profiler_test(
        testName="WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest ./tests/ttnn/tracy/test_dispatch_profiler.py::test_with_ops",
        setupAutoExtract=True,
    )
    for device, deviceData in devicesData["data"]["devices"].items():
        for ref, counts in REF_COUNT_DICT.items():
            if ref in deviceData["cores"]["DEVICE"]["analysis"].keys():
                assert (
                    deviceData["cores"]["DEVICE"]["analysis"][ref]["stats"]["Count"] in counts
                ), "Wrong ethernet dispatch zone count"

    devicesData = run_device_profiler_test(
        testName="WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml pytest ./tests/ttnn/tracy/test_dispatch_profiler.py::test_all_devices",
        setupAutoExtract=True,
    )
    for device, deviceData in devicesData["data"]["devices"].items():
        for ref, counts in REF_COUNT_DICT.items():
            if ref in deviceData["cores"]["DEVICE"]["analysis"].keys():
                assert (
                    deviceData["cores"]["DEVICE"]["analysis"][ref]["stats"]["Count"] in counts
                ), "Wrong ethernet dispatch zone count"
    os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"


@skip_for_grayskull()
def test_profiler_host_device_sync():
    TOLERANCE = 0.1

    os.environ["TT_METAL_PROFILER_SYNC"] = "1"
    syncInfoFile = PROFILER_LOGS_DIR / PROFILER_HOST_DEVICE_SYNC_INFO

    deviceData = run_device_profiler_test(testName="pytest ./tests/ttnn/tracy/test_profiler_sync.py::test_all_devices")
    reportedFreq = deviceData["data"]["deviceInfo"]["freq"] * 1e6
    assert os.path.isfile(syncInfoFile)

    syncinfoDF = pd.read_csv(syncInfoFile)
    devices = sorted(syncinfoDF["device id"].unique())
    for device in devices:
        deviceFreq = syncinfoDF[syncinfoDF["device id"] == device].iloc[-1]["frequency"]
        if not np.isnan(deviceFreq):  # host sync entry
            freq = float(deviceFreq) * 1e9

            assert freq < (reportedFreq * (1 + TOLERANCE)), f"Frequency {freq} is too large on device {device}"
            assert freq > (reportedFreq * (1 - TOLERANCE)), f"Frequency {freq} is too small on device {device}"
        else:  # device sync entry
            deviceFreqRatio = syncinfoDF[syncinfoDF["device id"] == device].iloc[-1]["device_frequency_ratio"]
            assert deviceFreqRatio < (
                1 + TOLERANCE
            ), f"Frequency ratio {deviceFreqRatio} is too large on device {device}"
            assert deviceFreqRatio > (
                1 - TOLERANCE
            ), f"Frequency ratio {deviceFreqRatio} is too small on device {device}"

    deviceData = run_device_profiler_test(testName="pytest ./tests/ttnn/tracy/test_profiler_sync.py::test_with_ops")
    reportedFreq = deviceData["data"]["deviceInfo"]["freq"] * 1e6
    assert os.path.isfile(syncInfoFile)

    syncinfoDF = pd.read_csv(syncInfoFile)
    devices = sorted(syncinfoDF["device id"].unique())
    for device in devices:
        deviceFreq = syncinfoDF[syncinfoDF["device id"] == device].iloc[-1]["frequency"]
        if not np.isnan(deviceFreq):  # host sync entry
            freq = float(deviceFreq) * 1e9

            assert freq < (reportedFreq * (1 + TOLERANCE)), f"Frequency {freq} is too large on device {device}"
            assert freq > (reportedFreq * (1 - TOLERANCE)), f"Frequency {freq} is too small on device {device}"

    os.environ["TT_METAL_PROFILER_SYNC"] = "0"


def test_timestamped_events():
    OP_COUNT = 2
    RISC_COUNT = 5
    ZONE_COUNT = 100
    ERISC_COUNTS = [0, 1, 5]
    TENSIX_COUNTS = [72, 64, 56]

    COMBO_COUNTS = []
    for T in TENSIX_COUNTS:
        for E in ERISC_COUNTS:
            COMBO_COUNTS.append((T, E))

    REF_COUNT_DICT = {
        "grayskull": [108 * OP_COUNT * RISC_COUNT * ZONE_COUNT, 88 * OP_COUNT * RISC_COUNT * ZONE_COUNT],
        "wormhole_b0": [(T * RISC_COUNT + E) * OP_COUNT * ZONE_COUNT for T, E in COMBO_COUNTS],
    }
    REF_ERISC_COUNT = {
        "wormhole_b0": [C * OP_COUNT * ZONE_COUNT for C in ERISC_COUNTS],
    }

    ENV_VAR_ARCH_NAME = os.getenv("ARCH_NAME")
    assert ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys()

    devicesData = run_device_profiler_test(setupAutoExtract=True)

    if ENV_VAR_ARCH_NAME in REF_ERISC_COUNT.keys():
        eventCount = len(
            devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["riscs"]["TENSIX"]["events"]["erisc_events"]
        )
        assert eventCount in REF_ERISC_COUNT[ENV_VAR_ARCH_NAME], "Wrong erisc event count"

    if ENV_VAR_ARCH_NAME in REF_COUNT_DICT.keys():
        eventCount = len(
            devicesData["data"]["devices"]["0"]["cores"]["DEVICE"]["riscs"]["TENSIX"]["events"]["all_events"]
        )
        assert eventCount in REF_COUNT_DICT[ENV_VAR_ARCH_NAME], "Wrong event count"


def test_sub_device_profiler():
    ARCH_NAME = os.getenv("ARCH_NAME")
    run_gtest_profiler_test(
        "./build/test/tt_metal/unit_tests_dispatch" + "_" + ARCH_NAME,
        "CommandQueueSingleCardFixture.TensixTestSubDeviceBasicPrograms",
    )
    os.environ["TT_METAL_PROFILER_SYNC"] = "1"
    run_gtest_profiler_test(
        "./build/test/tt_metal/unit_tests_dispatch" + "_" + ARCH_NAME,
        "CommandQueueSingleCardFixture.TensixActiveEthTestSubDeviceBasicEthPrograms",
    )
    os.environ["TT_METAL_PROFILER_SYNC"] = "0"
    run_gtest_profiler_test(
        "./build/test/tt_metal/unit_tests_dispatch" + "_" + ARCH_NAME,
        "CommandQueueSingleCardTraceFixture.TensixTestSubDeviceTraceBasicPrograms",
    )
