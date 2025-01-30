#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2024 Tenstorrent, Inc. All rights reserved.
#
# This script is based on `xrtdeps.sh` from the Xilinx XRT project.
# Original source: https://github.com/Xilinx/XRT/blob/master/src/runtime_src/tools/scripts/xrtdeps.sh
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FLAVOR=`grep '^ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"'`
VERSION=`grep '^VERSION_ID=' /etc/os-release | awk -F= '{print $2}' | tr -d '"'`
MAJOR=${VERSION%.*}
ARCH=`uname -m`

if [ $FLAVOR != "ubuntu" ]; then
    echo "Error: Only Ubuntu is supported"
    exit 1
fi

if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root. Please use sudo."
    usage
fi

usage()
{
    echo "Usage: sudo ./install_dependencies.sh [options]"
    echo
    echo "[--help, -h]                List this help"
    echo "[--validate, -v]            Validate that required packages are installed"
    echo "[--docker, -d]              Specialize execution for docker"
    echo "[--mode, -m <mode>]         Select installation mode: runtime, build, baremetal"
    exit 1
}

validate=0
docker=0
mode="baremetal"

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            ;;
        --validate|-v)
            validate=1
            shift
            ;;
        --docker|-d)
            docker=1
            shift
            ;;
	--mode|-m)
            mode="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# libc++ runtime dependency could eventually go away
# It is favored on Ubuntu20.04 for C++20 support

# At the time of this writing the following libraries are linked at runtime by sfpi cross compiler
# libmpc, libmfpr, libgmp, libz
# For the time being it will be assumed that these packages come from the base Ubuntu image

# Don't really need the dev package for libhwloc here, but this is simpler for now

ub_runtime_packages()
{
    UB_RUNTIME_LIST=(\
     python3-pip \
     libhwloc-dev \
     libc++1-17 \
     libc++abi1-17 \
    )
}

ub_buildtime_packages()
{
    UB_BUILDTIME_LIST=(\
     libpython3-dev \
     python3-pip \
     cmake \
     ninja-build
     libhwloc-dev \
     libc++-17-dev \
     libc++abi-17-dev \
    )
}

# Packages needed to setup a baremetal machine to build from source and run

ub_baremetal_packages() {
    ub_runtime_packages
    ub_buildtime_packages
    UB_BAREMETAL_LIST=("${UB_RUNTIME_LIST[@]}" "${UB_BUILDTIME_LIST[@]}")
}

update_package_list()
{
    if [ $FLAVOR == "ubuntu" ]; then
	case "$mode" in
            runtime)
                ub_runtime_packages
                PKG_LIST=("${UB_RUNTIME_LIST[@]}")
                ;;
            build)
                ub_buildtime_packages
                PKG_LIST=("${UB_BUILD_LIST[@]}")
                ;;
            baremetal)
                ub_baremetal_packages
                PKG_LIST=("${UB_BAREMETAL_LIST[@]}")
                ;;
            *)
                echo "Invalid mode: $mode"
                usage
                ;;
        esac
    fi
}

validate_packages()
{
    if [ $FLAVOR == "ubuntu" ]; then
        dpkg -l "${PKG_LIST[@]}"
    fi
}

prep_ubuntu()
{
    echo "Preparing ubuntu ..."
    # Update the list of available packages
    apt-get update
    apt install -y --no-install-recommends ca-certificates gpg wget
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null
    apt update
}

# We currently have an affinity to clang as it is more thoroughly tested in CI
# However g++-12 and later should also work

install_llvm() {
    LLVM_VERSION="17"
    echo "Checking if LLVM $LLVM_VERSION is already installed..."
    if command -v clang-$LLVM_VERSION &> /dev/null; then
        echo "LLVM $LLVM_VERSION is already installed. Skipping installation."
    else
        echo "Installing LLVM $LLVM_VERSION..."
        TEMP_DIR=$(mktemp -d)
        wget -P $TEMP_DIR https://apt.llvm.org/llvm.sh
        chmod u+x $TEMP_DIR/llvm.sh
        $TEMP_DIR/llvm.sh $LLVM_VERSION
        rm -rf "$TEMP_DIR"
    fi
}

# We don't really want to have this dependency
# This could be removed in the future

configure_hugepages() {
    TT_TOOLS_VERSION='1.1-5_all'
    echo "Installing Tenstorrent Hugepages Service $TT_TOOLS_VERSION..."
    TEMP_DIR=$(mktemp -d)
    wget -P $TEMP_DIR https://github.com/tenstorrent/tt-system-tools/releases/download/upstream%2F1.1/tenstorrent-tools_${TT_TOOLS_VERSION}.deb
    apt-get install $TEMP_DIR/tenstorrent-tools_${TT_TOOLS_VERSION}.deb
    systemctl enable --now tenstorrent-hugepages.service
    rm -rf "$TEMP_DIR"
}

install() {
    if [ $FLAVOR == "ubuntu" ]; then
        prep_ubuntu

        echo "Installing packages..."
        DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends "${PKG_LIST[@]}"

	case "$mode" in
            build)
                install_llvm
                ;;
            baremetal)
                install_llvm
                configure_hugepages
                ;;
        esac
    fi
}

update_package_list

if [ $validate == 1 ]; then
    validate_packages
else
    install
fi
