import numpy as np


# replace each pixel with its best output
def mat_absmin(mat):
    abs_index = mat_argmin(abs(mat))
    result = np.zeros(abs_index.shape)
    for row in range(abs_index.shape[0]):
        for col in range(abs_index.shape[1]):
            ind = abs_index[row][col]
            result[row][col] = mat[ind][row][col]
    return result


# return index of best kernel for each pixel
def mat_argmin(mat):
    mat_flat = mat.flatten()
    result_index = []
    step = len(mat[0].flatten())
    for i in range(step):
        result_index.append(np.argmin(mat_flat[i::step]))
    result_index = np.reshape(np.array(result_index), mat[0].shape)
    return result_index
