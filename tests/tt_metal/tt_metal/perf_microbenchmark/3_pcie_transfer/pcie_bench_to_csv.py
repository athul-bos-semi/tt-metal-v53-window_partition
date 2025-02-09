# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

#
# Capture the benchmark results json with this command
# then run this script to convert it to a csv suitable for a bar chart
# TT_METAL_LOGGER_LEVEL=FATAL ./build/test/tt_metal/perf_microbenchmark/3_pcie_transfer/pcie_bench_wormhole_b0 \
# --benchmark_out=bench_results.json --benchmark_out_format=json
#
import json
import sys
import pandas as pd


def get_filename() -> str:
    if len(sys.argv) != 2:
        print("Usage: python pcie_bench_to_csv.py <filename>")
        sys.exit(1)

    return str(sys.argv[1])


def Host_Write_HP_N_Readers(json_data):
    prefix = "Host_Write_HP_N_Readers_WithHostCopy"
    data_points = []
    for benchmark in json_data["benchmarks"]:
        name = benchmark["name"]
        if not name.startswith(prefix):
            continue

        data_points.append(
            {
                "total_size": benchmark["total_size"],
                "num_readers": benchmark["num_readers"],
                "dev_bytes_per_second": benchmark["dev_bandwidth_per_second"],
                "host_bytes_per_second": benchmark["bytes_per_second"],
                "page_size": benchmark["page_size"],
            }
        )

    df = pd.DataFrame(data_points)
    df = df.sort_values(by=["total_size", "page_size", "num_readers"], ascending=True)
    csv_str = df.to_csv(index=False)

    return csv_str


def Host_Write_HP_N_Threads(json_data):
    prefix = "Host_Write_HP_N_Threads"
    data_points = []
    for benchmark in json_data["benchmarks"]:
        if not benchmark["name"].startswith(prefix):
            continue

        data_points.append(
            {
                "total_size": benchmark["total_size"],
                "num_threads": benchmark["num_threads"],
                "host_bytes_per_second": benchmark["bytes_per_second"],
                "page_size": benchmark["page_size"],
            }
        )

    df = pd.DataFrame(data_points)
    df = df.sort_values(by=["total_size", "page_size", "num_threads"], ascending=True)
    csv_str = df.to_csv(index=False)

    return csv_str


def main():
    with open(get_filename(), "r") as fd:
        data = json.load(fd)
    csv_output = Host_Write_HP_N_Readers(data)

    # Print CSV output
    print(csv_output)


if __name__ == "__main__":
    main()
