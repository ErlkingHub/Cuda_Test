import numpy as np
from numba import cuda

@cuda.jit
def BinarySearch(accumulative_value, key, chrom_len):
    '''返回满足 = key的 个体染色体位置'''
    low = 0
    high = chrom_len - 1
    if (key > accumulative_value[high]):
        return high
    if (key < accumulative_value[low]):
        return low
    while (low <= high):
        mid = (low + high) / 2
