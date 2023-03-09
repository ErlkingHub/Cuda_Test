from time import time
from configg import *
from kmeans import *
import json
import numpy as np


'''
    打印聚类信息
'''
def printCentroid(arrayC, arrayCsum, arrayCnumpoint):
    for i in range(NUMBER_OF_CENTROIDS):
        print("[x={:6f}, y={:6f}, x_sum={:6f}, y_sum={:6f}, num_points={:d}]".format(
            arrayC[i, 0], arrayC[i, 1], arrayCsum[i, 0], arrayCsum[i, 1], arrayCnumpoint[i])
        )

    print('--------------------------------------------------')


'''
    实验代码，用于计时与重复实验
'''
def runKmeans(arrayP, arrayPclusters,
              arrayC, arrayCsum, arrayCnumpoint):

    # 开始计时
    start = time()

    for i in range(REPEAT):
        # 使用点对象中的前k个点初始化聚类
        for i1 in range(NUMBER_OF_CENTROIDS):
            arrayC[i1, 0] = arrayP[i1, 0]
            arrayC[i1, 1] = arrayP[i1, 1]

        arrayC, arrayCsum, arrayCnumpoint = kmeans(
            arrayP, arrayPclusters,
            arrayC, arrayCsum, arrayCnumpoint,
            NUMBER_OF_POINTS, NUMBER_OF_CENTROIDS
        )

        if i + 1 == REPEAT:
            printCentroid(arrayC, arrayCsum, arrayCnumpoint)

    # 结束计时
    end = time()
    total = (end - start) * 1000 / REPEAT

    print("Iterations: {:d}".format(ITERATIONS))
    print("Average Time: {:.4f} ms".format(total))


def main():
    # 从json文件中读取数据集
    with open("./points.json") as f:
        listPoints = list(map(lambda x: (x[0], x[1]), json.loads(f.read())))

    # 初始化变量
    arrayP = np.ones((NUMBER_OF_POINTS, 2), dtype=np.float32)
    arrayPclusters = np.ones(NUMBER_OF_POINTS, dtype=np.int32)

    arrayC = np.ones((NUMBER_OF_CENTROIDS, 2), dtype=np.float32)
    arrayCsum = np.ones((NUMBER_OF_CENTROIDS, 2), dtype=np.float32)
    arrayCnumpoint = np.ones(NUMBER_OF_CENTROIDS, dtype=np.int32)

    # 初始化点对象
    for i, d in enumerate(listPoints):
        arrayP[i, 0] = d[0]
        arrayP[i, 1] = d[1]

    runKmeans(arrayP, arrayPclusters,
              arrayC, arrayCsum, arrayCnumpoint)


if __name__ == '__main__':
    main()
