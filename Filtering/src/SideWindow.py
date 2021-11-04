import numpy as np
import cv2
import copy
import cv_joke as cv_jk
import scipy.signal as sci
import matrix as mt
import random
import time


def generate(base, r0, r1, c0, c1, value):  # set appointed area to value
    ret = base.copy()
    num_grids = (r1 - r0) * (c1 - c0)
    if value == 0:
        value = 1 / num_grids
    add = np.ones([r1 - r0, c1 - c0])
    add = add * value
    ret = ret + add
    return ret

'''Side Mean Filter实现'''


def SW_MeanFil(img, r, iteration=1):
    zeros = np.zeros([2 * r + 1, 2 * r + 1])
    kernel_L  = generate(zeros, 0, 2 * r + 1, 0, r + 1, 0)
    kernel_R  = generate(zeros, 0, 2 * r + 1, r, 2 * r + 1, 0)
    kernel_U  = generate(zeros, 0, r + 1, 0, 2 * r + 1, 0)
    kernel_D  = generate(zeros, r, 2 * r + 1, 0, r + 1, 0)
    kernel_NW = generate(zeros, 0, r + 1, 0, r + 1, 0)
    kernel_NE = generate(zeros, 0, r + 1, r, 2 * r + 1, 0)
    kernel_SW = generate(zeros, r, 2 * r + 1, 0, r + 1, 0)
    kernel_SE = generate(zeros, r, 2 * r + 1, r, 2 * r + 1, 0)

    kernels = [kernel_L, kernel_R, kernel_U, kernel_D,
               kernel_NW, kernel_NE, kernel_SW, kernel_SE]
    m = img.shape[0] + 2 * r
    n = img.shape[1] + 2 * r
    num_channels = img.shape[2]
    dis = np.zeros([8, m, n])
    ret = img.copy()

    for channel in range(num_channels):
        origin = np.pad(img[:, :, channel], )

    for ch in range(img.shape[2]):
        U = np.pad(img[:, :, ch], (r, r), 'edge')
        for i in range(iteration):
            for id, kernel in enumerate(kernels):
                conv2 = sci.correlate2d(U, kernel, 'same')
                dis[id] = conv2 - U
            U = U + mt.mat_absmin(dis)
        result[:, :, ch] = U[r:-r, r:-r]
    return result


'''滑动内核与矩阵星乘取中值'''


def mid_mult(img, kernel, r, start_offset, end_offset):
    result = []
    for row in range(start_offset[0], img.shape[0] - kernel.shape[0] + 1 - end_offset[0]):
        for col in range(start_offset[1], img.shape[1] - kernel.shape[1] + 1 - end_offset[1]):
            # img_roi = copy.deepcopy(img[row:row+kernel.shape[0],col:col+kernel.shape[1]])
            img_roi = img[row:row + kernel.shape[0], col:col + kernel.shape[1]]
            mid_tmp = np.median(img_roi * kernel)
            result.append(mid_tmp)
    result = np.reshape(np.array(result), (img.shape[0] - 2 * r, img.shape[1] - 2 * r))
    result = np.pad(result, (r, r), 'edge');
    return result


'''Side Median Filter实现'''


def s_medianfilter(img, radius, iteration=1):
    r = radius
    # 异型内核
    k_L = np.ones((2 * r + 1, r + 1))
    k_R = k_L
    k_U = k_L.T
    k_D = k_U
    k_NW = np.ones((r + 1, r + 1))
    k_NE = k_NW
    k_SW = k_NW
    k_SE = k_NW
    kernels = [k_L, k_R, k_U, k_D, k_NW, k_NE, k_SW, k_SE]
    start_offsets = [(0, 0), (0, r), (0, 0), (r, 0), (0, 0), (0, r), (r, 0), (r, r)]
    end_offsets = [(0, r), (0, 0), (r, 0), (0, 0), (r, r), (r, 0), (0, r), (0, 0)]

    m = img.shape[0] + 2 * r
    n = img.shape[1] + 2 * r
    dis = np.zeros([8, m, n]);
    result = copy.deepcopy(img)
    for ch in range(img.shape[2]):
        U = np.pad(img[:, :, ch], (r, r), 'edge');
        for i in range(iteration):
            for id in range(len(kernels)):
                mid_result = mid_mult(U, kernels[id], r, start_offsets[id], end_offsets[id])
                dis[id] = mid_result - U
            U = U + mt.mat_absmin(dis)
        result[:, :, ch] = U[r:-r, r:-r]
    return result


'''将内核的某一区域保留,其余置0'''


def zeros_kernel(kernel, size=(1, 1), loc=(0, 0)):
    kernel_tmp = np.zeros(kernel.shape)
    kernel_tmp[loc[0]:(loc[0] + size[0]), loc[1]:(loc[1] + size[1])] = kernel[loc[0]:(loc[0] + size[0]),
                                                                       loc[1]:(loc[1] + size[1])]
    kernel_tmp = kernel_tmp / np.sum(kernel_tmp)
    return kernel_tmp


'''Side Gaussian Filter实现'''


def s_gausfilter(img, radius, sigma=0, iteration=1):
    r = radius
    gaus_kernel = cv2.getGaussianKernel(2 * r + 1, sigma)  # sigma = ((n-1)*0.5 - 1)*0.3 + 0.8
    gaus_kernel = gaus_kernel.dot(gaus_kernel.T)
    gaus_kernel = gaus_kernel.astype(np.float)
    k_L = zeros_kernel(gaus_kernel, size=(2 * r + 1, r + 1), loc=(0, 0))
    k_R = zeros_kernel(gaus_kernel, size=(2 * r + 1, r + 1), loc=(0, r))
    K_U = k_L.T
    k_D = K_U[::-1]
    k_NW = zeros_kernel(gaus_kernel, size=(r + 1, r + 1), loc=(0, 0))
    k_NE = zeros_kernel(gaus_kernel, size=(r + 1, r + 1), loc=(0, r))
    k_SW = k_NW[::-1]
    k_SE = k_NE[::-1]
    kernels = [k_L, k_R, K_U, k_D, k_NW, k_NE, k_SW, k_SE]
    m = img.shape[0] + 2 * r
    n = img.shape[1] + 2 * r
    dis = np.zeros([8, m, n])
    result = copy.deepcopy(img)
    for ch in range(img.shape[2]):
        U = np.pad(img[:, :, ch], (r, r), 'edge')
        for i in range(iteration):
            for id, kernel in enumerate(kernels):
                conv2 = sci.correlate2d(U, kernel, 'same')
                dis[id] = conv2 - U
            U = U + mt.mat_absmin(dis)
        result[:, :, ch] = U[r:-r, r:-r]
    return result

