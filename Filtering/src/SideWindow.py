import matplotlib
import numpy as np
import scipy.signal as sci
from PIL import Image
import matrix as mt
import matplotlib.pyplot as plt


def generate(base, r0, r1, c0, c1, value):  # set appointed area to value
    ret = base.copy()
    num_grids = (r1 - r0) * (c1 - c0)
    if value == 0:
        value = 1 / num_grids
    add = np.ones([r1 - r0, c1 - c0])
    add = add * value
    ret[r0:r1, c0:c1] = add
    return ret


# complete in detail, with something wrong
def convolve(kernels, radius, row, col, origin):
    ret_pixel = origin[row + radius][col + radius]
    base = ret_pixel
    min_diff = 2147483647
    for kernel in kernels:
        cur_output = 0
        for i in range(0, 2 * radius + 1):
            for j in range(0, 2 * radius + 1):
                cur_output += origin[row + i][col + j] * kernel[i][j]
        cur_diff = abs(cur_output - base)
        if cur_diff < min_diff:
            ret_pixel = cur_output
            min_diff = cur_diff
    return ret_pixel


# Mean filter with side window
# Main usage: image smoothing, denoising, maybe cartoon
def SW_MeanFil(img, r, iteration):
    zeros = np.zeros([2 * r + 1, 2 * r + 1])
    # set all types of kernel to the same size by setting unrelated weights to zero, to simplify calculation
    kernel_L = generate(zeros, 0, 2 * r + 1, 0, r + 1, 0)
    kernel_R = generate(zeros, 0, 2 * r + 1, r, 2 * r + 1, 0)
    kernel_U = generate(zeros, 0, r + 1, 0, 2 * r + 1, 0)
    kernel_D = generate(zeros, r, 2 * r + 1, 0, 2 * r + 1, 0)
    kernel_NW = generate(zeros, 0, r + 1, 0, r + 1, 0)
    kernel_NE = generate(zeros, 0, r + 1, r, 2 * r + 1, 0)
    kernel_SW = generate(zeros, r, 2 * r + 1, 0, r + 1, 0)
    kernel_SE = generate(zeros, r, 2 * r + 1, r, 2 * r + 1, 0)

    kernels = [kernel_L, kernel_R, kernel_U, kernel_D,
               kernel_NW, kernel_NE, kernel_SW, kernel_SE]

    # for padding
    m = img.shape[0] + 2 * r
    n = img.shape[1] + 2 * r
    num_channels = img.shape[2]

    ret = img.copy()
    # for it in range(iteration):
    #     for channel in range(num_channels):
    #         origin = np.pad(ret[:, :, channel], (r, r), mode='constant', constant_values=0)  # padding by zero
    #         for row in range(img.shape[0]):
    #             for col in range(img.shape[1]):
    #                 ret[row][col] = convolve(kernels, r, row, col, origin)

    d = np.zeros([8, m, n])  # cache of differences between output and input, 8 for 8 kernels
    for channel in range(num_channels):
        # origin = np.pad(img[:, :, channel], (r, r), mode='edge')  # padding by values at edge; which let pixels near edge influenced much more by global image edge
        origin = np.pad(ret[:, :, channel], (r, r), mode='constant', constant_values=0)  # padding by zero
        for it in range(iteration):
            for i, kernel in enumerate(kernels):
                # learned some convolution operation for speedup
                cur_output = sci.correlate2d(origin, kernel, 'same')
                d[i] = cur_output - origin
            origin = origin + mt.mat_absmin(d)
            ret[:, :, channel] = origin[r:-r, r:-r]
    return ret


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


def SW_MidFil(img, r, iteration=1):
    k_L = np.ones([2 * r + 1, r + 1])
    k_R = k_L
    k_U = k_L.T
    k_D = k_U
    k_NW = np.ones((r + 1, r + 1))
    k_NE = k_NW
    k_SW = k_NW
    k_SE = k_NW
    # due to difference between mid and mean filter, size of kernels for median filter are different
    kernel_L = np.ones([2 * r + 1, r + 1])
    kernel_R = kernel_L  # same in appearance while different during use
    kernel_U = kernel_L.transpose()
    kernel_D = kernel_U
    kernel_NW = kernel_NE = kernel_SW = kernel_SE = np.ones([r + 1, r + 1])

    kernels = [kernel_L, kernel_R, kernel_U, kernel_D,
               kernel_NW, kernel_NE, kernel_SW, kernel_SE]

    start_offsets = [(0, 0), (0, r), (0, 0), (r, 0), (0, 0), (0, r), (r, 0), (r, r)]
    end_offsets = [(0, r), (0, 0), (r, 0), (0, 0), (r, r), (r, 0), (0, r), (0, 0)]

    # for padding
    m = img.shape[0] + 2 * r
    n = img.shape[1] + 2 * r
    num_channels = img.shape[2]
    ret = img.copy()
    d = np.zeros([8, m, n])  # cache of differences between output and input, 8 for 8 kernels

    for channel in range(num_channels):
        # origin = np.pad(img[:, :, channel], (r, r), mode='constant', constant_values=0)
        # origin = np.pad(img[:, :, channel], (r, r), mode='edge')
        origin = np.pad(img[:, :, channel], (r, r), mode='median')
        for it in range(iteration):
            for i, kernel in enumerate(kernels):
                cur_output = sci.correlate2d(origin, kernel, 'same')
                d[i] = cur_output - origin
            origin = origin + mt.mat_absmin(d)
            ret[:, :, channel] = origin[r:-r, r:-r]


    for ch in range(img.shape[2]):
        U = ;
        for i in range(iteration):
            for id in range(len(kernels)):
                mid_result = mid_mult(U, kernels[id], r, start_offsets[id], end_offsets[id])
                dis[id] = mid_result - U
            U = U + mt.mat_absmin(dis)
        result[:, :, ch] = U[r:-r, r:-r]
    return result


if __name__ == '__main__':
    img_path = "/home/melodies/Downloads/pns_panda.jpg"
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = np.asarray(img)  # format: rows, cols, channels
    output = SW_MeanFil(img, 3, 3)
    img = Image.fromarray(output)
    img.show()
    img.save("/home/melodies/Downloads/pns_panda_output.jpg")
    input()
