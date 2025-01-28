# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
from tests.sweep_framework.framework.sweeps_logger import sweeps_logger as logger

GRAYSKULL_ARGS = ["-tr", "all"]
LEGACY_WORMHOLE_ARGS = ["-wr", "all"]
WORMHOLE_ARGS = ["-r"]

RESET_OVERRIDE = os.getenv("TT_SMI_RESET_COMMAND")


def run_tt_smi(arch: str):
    if RESET_OVERRIDE is not None:
        smi_process = subprocess.run(RESET_OVERRIDE, shell=True)
        if smi_process.returncode == 0:
            logger.info("TT-SMI Reset Complete Successfully")
            return
        else:
            raise Exception(f"SWEEPS: TT-SMI Reset Failed with Exit Code: {smi_process.returncode}")
        return

    if arch not in ["grayskull", "wormhole_b0"]:
        raise Exception(f"SWEEPS: Unsupported Architecture for TT-SMI Reset: {arch}")

    smi_options = [
        "tt-smi",
        "tt-smi-metal",
        "/home/software/syseng/gs/tt-smi" if arch == "grayskull" else "/home/software/syseng/wh/tt-smi",
    ]
    args = GRAYSKULL_ARGS if arch == "grayskull" else WORMHOLE_ARGS

    # Potential implementation to use the pre-defined reset script on each CI runner. Not in use because of the time and extra functions it has. (hugepages check, etc.)
    # if os.getenv("CI") == "true":
    #     smi_process = subprocess.run(f"/opt/tt_metal_infra/scripts/ci/{arch}/reset.sh", shell=True, capture_output=True)
    #     if smi_process.returncode == 0:
    #         print("SWEEPS: TT-SMI Reset Complete Successfully")
    #         return
    #     else:
    #         raise Exception(f"SWEEPS: TT-SMI Reset Failed with Exit Code: {smi_process.returncode}")

    for smi_option in smi_options:
        executable = shutil.which(smi_option)
        if executable is not None:
            # Corner case for newer version of tt-smi, -tr and -wr are removed on this version (tt-smi-metal). Default device 0, if needed use TT_SMI_RESET_COMMAND override.
            if smi_option == "tt-smi-metal":
                args = ["-r", "0"]
            elif arch == "grayskull":
                args = GRAYSKULL_ARGS
            elif arch == "wormhole_b0":
                smi_process = subprocess.run([executable, "-v"], capture_output=True, text=True)
                smi_version = smi_process.stdout.strip()
                if not smi_version.startswith("3.0"):
                    args = LEGACY_WORMHOLE_ARGS
                else:
                    args = WORMHOLE_ARGS
                    smi_process = subprocess.run([executable, "-g", ".reset.json"])
                    import json
                    with open(".reset.json", "r") as f:
                        reset_json = json.load(f)
                        card_id = reset_json["wh_link_reset"]["pci_index"]
                        if len(card_id) < 1:
                            raise Exception(f"SWEEPS: TT-SMI Reset Failed to Find Card ID.")
                        args.append(str(card_id[0]))
                    subprocess.run(["rm", "-f", ".reset.json"])
            smi_process = subprocess.run([executable, *args])
            if smi_process.returncode == 0:
                logger.info("SWEEPS: TT-SMI Reset Complete Successfully")
                return
            else:
                raise Exception(f"SWEEPS: TT-SMI Reset Failed with Exit Code: {smi_process.returncode}")

    raise Exception("SWEEPS: Unable to Locate TT-SMI Executable")
