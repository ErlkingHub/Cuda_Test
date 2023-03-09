# from math import sqrt
from configg import *
import numba
from numba import cuda
import numpy as np

@numba.jit(nopython=True)
def groupByCluster(arrayP, arrayPcluster,
                   arrayC,
                   num_points, num_centroids, tmp_result):
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
    if idx < num_points:
        dx_list = arrayP[idx,0] - arrayC[:,0]
        dy_list = arrayP[idx,1] - arrayC[:,1]
    tmp_result = np.multiply(dx_list, dx_list) + np.multiply(dy_list, dy_list)
    return tmp_result

@numba.jit(nopython=True)
def groupByCluster2(tmp_result, arrayPcluster):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    minor_distance = -1
    my_disList = np.sqrt(tmp_result)
    if idx < len(arrayPcluster):
        for i in range(10):
            if minor_distance > my_disList[idx] or minor_distance == -1:
                    minor_distance = my_disList[idx]
                    arrayPcluster[idx] = i
    return arrayPcluster
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
    for i in range(num_centroids):
        arrayCsum[i, 0] = 0
        arrayCsum[i, 1] = 0
        arrayCnumpoint[i] = 0

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



def kmeans(arrayP, arrayPcluster,
           arrayC, arrayCsum, arrayCnumpoint,
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
    tmp_result = cuda.device_array(num_centroids)
    for i in range(ITERATIONS):
        groupByCluster(
            arrayP, arrayPcluster,
            arrayC,
            num_points, num_centroids, tmp_result
        )
        groupByCluster2(tmp_result, arrayPcluster)

        calCentroidsSum(
            arrayP, arrayPcluster,
            arrayCsum, arrayCnumpoint,
            num_points, num_centroids
        )

        updateCentroids(
            arrayC, arrayCsum, arrayCnumpoint,
            num_centroids
        )

    return arrayC, arrayCsum, arrayCnumpoint
