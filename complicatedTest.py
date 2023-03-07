''' This is a more complicated trial for numba.cuda.
    Aiming to make a comparision for cuda-C'''

import numpy as np
from numba import cuda,jit,njit

# 随机数生成
import random
from numba.cuda.random import init_xoroshiro128p_states, create_xoroshiro128p_states, xoroshiro128p_uniform_float32

@jit(nopython = True) #  = @njit
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
        if accumulative_value[mid] == key:
            return mid
        if accumulative_value[mid] < key and accumulative_value[mid +1]>=key:
            return mid+1
        if accumulative_value[mid] > key:
            high = mid-1
        if accumulative_value[mid] < key:
            low = mid+1
    return -1



@njit
def init_population(tseed, population_gpu, item_gpu, chrom_len, population_count_one_gpu, rand, one_twu_percent_gpu):
    id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    state = create_xoroshiro128p_states(1, seed=tseed)
    xoroshiro128p_uniform_float32(rng_states, thread_id)
    
    print(id)



if __name__ == '__main__':
    tseed = 21
    random.seed(tseed)