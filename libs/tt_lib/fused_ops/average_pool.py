import tt_lib as ttl

from typing import Union, List


def run_avg_pool_on_device_wrapper(device):
    def average_pool_2d(x):
        out = ttl.tensor.average_pool_2d(x)
        return out

    return average_pool_2d
