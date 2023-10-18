from numba import cuda, float32
import numpy as np
import time
import math


def tic():
    # Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
        return time.time() - startTime_for_tictoc
    else:
        print ("Toc: start time not set")


def multiply_matrix(A, B):
    tic()
    shape = int(len(A)), int(len(B[0]))
    if len(A) < len(B):
        exception = "Иди учи линейную алгебру, такие матрицы нельзя перемножить"
        return exception
    result = np.zeros(shape)
    for i in range(len(A)):
        # iterate through columns of B
        for j in range(len(B[0])):
            # iterate through rows of B
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    running_time = toc()
    return running_time


# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 # do the computation


@cuda.jit
def matmul_cuda(A, B):
    """Perform square matrix multiplication of C = A * B
    """
    print("+++++++++++++")
    shape_of_mat = int(len(A)), int(len(B[0]))
    if len(A) < len(B):
        exception = "Иди учи линейную алгебру, такие матрицы нельзя перемножить"
        return exception
    C = np.zeros(shape_of_mat)
    print("+++++++++++++")
    i, j = cuda.grid(2)
    print("+++++++++++++")
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp
    #threadsperblock = shape_of_mat
    #blockspergrid = math.ceil(С.shape[0] / threadsperblock)
    print(C)


def mat_generator(size_x, size_y, int_range=10):
    array = np.random.randint(int_range, size=(size_y, size_x))
    # print(array)
    return array


if __name__ == '__main__':
    SIZE = 3
    X = mat_generator(SIZE, SIZE)
    Y = mat_generator(SIZE, SIZE)
    print(multiply_matrix(X, Y))
    print(np.matmul(X, Y))
    print(matmul_cuda(X, Y))