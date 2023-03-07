''' This is a more complicated trial for numba.cuda.
    Aiming to make a comparision for cuda-C'''

import numpy as np
from numba import cuda,jit,njit

# 随机数生成
import random
from numba.cuda.random import init_xoroshiro128p_states, create_xoroshiro128p_states, xoroshiro128p_uniform_float32


# Global const Variables
tseed = 21
random.seed(tseed)
rng_states = create_xoroshiro128p_states(1, seed=tseed)

minUtility = 1100000
pop_size = 100

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
def init_population(population_gpu, chrom_len, population_count_one_gpu, one_twu_percent_gpu):
    ''' 初始化种群'''
    id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    number = xoroshiro128p_uniform_float32(rng_states, id)
    choosed_id = BinarySearch(one_twu_percent_gpu, number, chrom_len)
    x = choosed_id
    population_gpu[id * chrom_len + x] = 1
    population_count_one_gpu[id] = 1
    print(id)

@njit
def printf_kernel(database_gpu, eachline_length_gpu, start_position_gpu):
    for i in range(10):
        print("%d,%d::", i, start_position_gpu[i])
        for j in range(eachline_length_gpu[i]):
            print("%d ", database_gpu[start_position_gpu[i] + j])
        print("\n")

@njit
def printf_kernel1(frequent_length_gpu):
    print("%d,", frequent_length_gpu[0])

@njit
def printf_kernel2(frequent_length_gpu):
    for i in range(10):
        print("%d\n", frequent_length_gpu[i])

@njit
def printf_kernel4(population_gpu, chrom_len):
    for i in range(10):
        for j in range(chrom_len):
            if population_gpu[i * chrom_len + j] == 1:
                print("(%d) ", j, population_gpu[i * chrom_len + j])
        print("\n")

@njit
def calculate_fitness(population_gpu, database_gpu, database_utility_gpu, population_count_one_gpu, fitness_gpu, database_length, eachline_length_gpu, start_position_gpu, chrom_len):
    # ix = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.y
    ttid = tid + cuda.blockDim.x * cuda.blockIdx.x
    iter1 = 0 # 计算迭代次数

    num1 = 1 
    if ttid * num1 + num1 - 1 < database_length:
        iter1 = num1
    else:
        iter1 = database_length - ttid * num1 + 1
    tmp_fitness = 0
    for ii in range (iter1):
        k = 0
        tmp = 0
        for i in range(eachline_length_gpu[ttid * num1 + ii]):
            num = start_position_gpu[ttid * num1 + ii] + i
            if population_gpu[bid * chrom_len + database_gpu[num]] == 1:
                k+=1
                tmp += database_utility_gpu[num]
        if k == population_count_one_gpu[bid]:
            tmp_fitness += tmp

        # Supported on int32, int64, uint32, uint64, float32, float64 operands only.
    cuda.atomic.add(fitness_gpu[bid], tmp_fitness)

@njit
def calculate_result(fitness_gpu, result_gpu, result_length, frequent_item_gpu, frequent_length_gpu, population_gpu, chrom_len, MAX_ITER):
    for i in range(pop_size):
        fitness = fitness_gpu[i]
        if fitness >= minUtility:
            flag = False
            for j in range(result_length[0]):
                if result_gpu[j] == fitness:
                    flag = True
                    break
            if flag == True:
                for k in range(chrom_len):
                    if population_gpu[i * chrom_len + k] == 1:
                        flag1 = False
                        for m in range(frequent_length_gpu[0]):
                            if frequent_item_gpu[m] == k:
                                flag1 = True
                                break
                        if flag1 == True:
                            frequent_item_gpu[frequent_length_gpu[0]] = k
                            frequent_length_gpu[0] +=1
                result_gpu[result_length[0]] = fitness
                result_length[0] += 1

@njit
def mutation(population_gpu1, chrom_len, population_count_one_gpu1, one_twu_percent_gpu):
    ix = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    number = xoroshiro128p_uniform_float32(rng_states, ix)
    number1 = xoroshiro128p_uniform_float32(rng_states, ix)

    choosed_id = BinarySearch(one_twu_percent_gpu, number, chrom_len)
    choosed_id1 = BinarySearch(one_twu_percent_gpu, number1, chrom_len)
    x = choosed_id
    x1 = choosed_id
    x = x if x < x1 else x1 # <==> x = x < x1 ? x : x1
    ttid = id * chrom_len + x
    if population_gpu1[ttid] == 1:
        population_gpu1[ttid] = 0
        population_count_one_gpu1[id] -= 1
    else:
        population_gpu1[ttid] = 1
        population_count_one_gpu1[id] += 1

@njit
def kernel1(population_gpu, population_gpu1, population_gpu2, population_gpu3, population_count_one_gpu, population_count_one_gpu1, population_count_one_gpu2, population_count_one_gpu3, chrom_len, fitness_gpu, fitness_gpu1, fitness_gpu2, fitness_gpu3, rand, MAX_ITER):
    # ix = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.y
    ttid = tid + cuda.blockDim.x * cuda.blockIdx.x
    if ttid >= chrom_len:
        return
    tmp_num = 0
    share = cuda.shared.array(7, int)
    share[0] = fitness_gpu[bid]
    share[1] = fitness_gpu1[bid]
    share[2] = fitness_gpu2[bid]
    share[3] = fitness_gpu3[bid]
    for i in range(4):
        if share[i] > share[i - 1]:
            tmp_num = i
    index = bid * chrom_len + ttid
    index1 = ((bid + 1) % pop_size) * chrom_len + ttid
    if tmp_num == 1:
        population_gpu[index] = population_gpu1[index]
        if ttid == 0:
            fitness_gpu[bid] = fitness_gpu1[bid]
            population_count_one_gpu[bid] = population_count_one_gpu1[bid]
    elif (tmp_num == 2):
        population_gpu[index] = population_gpu2[index]
        if ttid == 0:
            fitness_gpu[bid] = fitness_gpu2[bid]
            population_count_one_gpu[bid] = population_count_one_gpu2[bid]
    elif (tmp_num == 3):
        population_gpu[index] = population_gpu3[index]
        if ttid == 0:
            fitness_gpu[bid] = fitness_gpu3[bid]
            population_count_one_gpu[bid] = population_count_one_gpu3[bid]
    # elif (tmp_num == 4):
    #     population_gpu[index] = population_gpu2[index]
    #     if ttid == 0:
    #         fitness_gpu[bid] = fitness_gpu2[bid]
    #         population_count_one_gpu[bid] = population_count_one_gpu2[bid]


@njit
def calculate_result_one(eachitem_twu_gpu,result_gpu,result_length, chrom_len):
    for i in range(chrom_len):
        if (eachitem_twu_gpu[i] >= minUtility):
            result_gpu[result_length[0]] = eachitem_twu_gpu[i]
            result_length[0] += 1
    print("%d,", result_length[0])


@njit
def parrallel_map_process(database_gpu,hash_sort_gpu,eachline_length_gpu,start_position_gpu,database_length):
    id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tid = cuda.threadIdx.x
    bid = cuda.blockIdx.y
    if id >= database_length:
        return
    start = start_position_gpu[id]
    for i in range(eachline_length_gpu[id]):
        database_gpu[start + i] = hash_sort_gpu[database_gpu[start + i]]


if __name__ == '__main__':
    print('Begin!!! \n')
