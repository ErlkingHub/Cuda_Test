# from math import sqrt
import math
from time import time
from configg import *
import numba
from numba import cuda
import numpy as np
from math import ceil

@cuda.jit()
def groupByCluster(arrayP, arrayPcluster,
                   arrayC, arrayCsum, arrayCnumpoint,
                   num_points, num_centroids):
    '''
        分配聚类
        arrayP        -> (10w,2)
        arrayPcluster -> (10w,)
        arrayC        -> (10,2)
        num_points = 10w
        num_centroids = 10
    '''
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    # x = cuda.threadIdx.x
    # bx = cuda.blockIdx.x
    # bdx = cuda.blockDim.x
    minor_distance = -1
    if idx < num_points:
        for i in range(num_centroids):
            dx_list = arrayP[idx,0] - arrayC[i,0]
            dy_list = arrayP[idx,1] - arrayC[i,1]
        # tmp_result = dx_list * dx_list + dy_list * dy_list
            my_disList = math.sqrt(dx_list * dx_list + dy_list * dy_list)
        # for i in range(num_centroids):
            if minor_distance > my_disList or minor_distance == -1:
                    minor_distance = my_disList
                    arrayPcluster[idx] = i
        
        # calCentroidsSum
        ci = arrayPcluster[idx]
        arrayCsum[ci, 0] += arrayP[idx, 0]
        arrayCsum[ci, 1] += arrayP[idx, 1]
        arrayCnumpoint[ci] += 1
        
        # updateCentroids
        # return arrayPcluster
    # return tmp_result

@cuda.jit
def groupByCluster2(arrayPcluster):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    minor_distance = -1
    
    # if idx < len(arrayPcluster):
        
    
    # for i0 in range(num_points):
    #     # 使用负数初始化当前聚类的最短距离
    #     minor_distance = -1
    #     for i1 in range(num_centroids):
    #         # 计算当前聚类均值点与点对象的距离
    #         dx = arrayP[i0, 0] - arrayC[i1, 0]
    #         dy = arrayP[i0, 1] - arrayC[i1, 1]
    #         my_distance = np.sqrt(dx * dx + dy * dy)
    #         # 假设当前距离的距离小于记录的距离，或记录距离为初始值则更新距离
    #         if minor_distance > my_distance or minor_distance == -1:
    #             minor_distance = my_distance
    #             arrayPcluster[i0] = i1
    return arrayPcluster



@numba.jit(nopython=True)
def calCentroidsSum(arrayP, arrayPcluster,
                    arrayCsum, arrayCnumpoint,
                    num_points, num_centroids):
    '''
    计算聚类总值
    arrayP        -> (10w,2)
    arrayPcluster -> (10w,)
    arrayCsum     -> (10,2)
    arrayCnumpoint-> (10,)
    num_points = 10w
    num_centroids = 10
    '''
    # 初始化聚类的总值信息
    # for i in range(num_centroids):
    #     arrayCsum[i, 0] = 0
    #     arrayCsum[i, 1] = 0
    #     arrayCnumpoint[i] = 0

    # 根据每个点对象所在的聚类，对聚类的总值信息进行更新
    for i in range(num_points):
        ci = arrayPcluster[i]
        arrayCsum[ci, 0] += arrayP[i, 0]
        arrayCsum[ci, 1] += arrayP[i, 1]
        arrayCnumpoint[ci] += 1

    return arrayCsum, arrayCnumpoint



@numba.jit(nopython=True)
def updateCentroids(arrayC, arrayCsum, arrayCnumpoint,
                    num_centroids):
    '''
    计算聚类均值
    arrayC        -> (10,2)
    arrayCsum     -> (10,2)
    arrayCnumpoint-> (10,)
    num_points = 10w
    num_centroids = 10
    '''
    
    for i in range(num_centroids):
        # 对已经计算好总值信息的聚类，计算其均值信息
        arrayC[i, 0] = arrayCsum[i, 0] / arrayCnumpoint[i]
        arrayC[i, 1] = arrayCsum[i, 1] / arrayCnumpoint[i]



def kmeans(arrayP_host, arrayPcluster_host,
           arrayC_host, arrayCsum_host, arrayCnumpoint_host,
           num_points, num_centroids):
    '''
        kmeans辅助代码
        arrayP        -> (10w,2)
        arrayPcluster -> (10w,)
        arrayC        -> (10,2)
        arrayCsum     -> (10,2)
        arrayCnumpoint-> (10,)
        num_points = 10w
        num_centroids = 10
    '''
    
    arrayP = cuda.to_device(arrayP_host)
    arrayPcluster = cuda.to_device(arrayPcluster_host)
    arrayC = cuda.to_device(arrayC_host)
    arrayCsum = cuda.to_device(arrayCsum_host)
    arrayCnumpoint = cuda.to_device(arrayCnumpoint_host)

    # start = time()
    
    # print(threads_per_block)
    # print(blocks_per_grid)
    # print(blocks_per_grid* threads_per_block)

    # tmp_result = cuda.device_array(num_centroids)

    for i in range(ITERATIONS):
        # blocks_per_grid = 2048
        # threads_per_block = ceil(num_points / blocks_per_grid)
        threads_per_block = 256
        blocks_per_grid = ceil(num_points / threads_per_block)
        groupByCluster[blocks_per_grid, threads_per_block](
            arrayP, arrayPcluster,
            arrayC, arrayCsum, arrayCnumpoint,
            num_points, num_centroids
        )
        # groupByCluster2[blocks_per_grid, threads_per_block](tmp_result, arrayPcluster)
        cuda.synchronize()
        # print(arrayPcluster[0:100])
        
        # calCentroidsSum(
        #     arrayP, arrayPcluster,
        #     arrayCsum, arrayCnumpoint,
        #     num_points, num_centroids
        # )
        arrayC_h = arrayC.copy_to_host()
        arrayCsum_h = arrayCsum.copy_to_host()
        arrayCnumpoint_h = arrayCnumpoint.copy_to_host()
        updateCentroids(
            arrayC_h, arrayCsum_h, arrayCnumpoint_h,
            num_centroids
        )
    return arrayC, arrayCsum, arrayCnumpoint
