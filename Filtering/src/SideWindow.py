import matplotlib
import numpy as np
import scipy.signal as sci
from PIL import Image
import matrix as mt
import matplotlib.pyplot as plt


# set appointed area to value
def generate(base, r0, r1, c0, c1, value):
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


def replace_mid(origin, kernel, r):
    ret = origin.copy()
    for row in range(r, -r):
        for col in range(r, -r):
            tmp = origin[row + kernel[2][0]:row + kernel[2][1], col + kernel[2][2]:col + kernel[2][3]]
            ret[row][col] = np.median(tmp)
    return ret


# 改了图片格式（HWC->CHW）
def SW_MidFil(img, r, iteration=1):
    # due to difference between mid and mean filter, size of kernels for median filter are different
    kernel_L  = [2 * r + 1, r + 1, [-r, r, -r, 0]]  # [rows, cols, [r0, r1, c0, c1]]
    kernel_R  = [2 * r + 1, r + 1, [-r, r, 0, r]]
    kernel_U  = [r + 1, 2 * r + 1, [-r, 0, -r, r]]
    kernel_D  = [r + 1, 2 * r + 1, [0, r, -r, r]]
    kernel_NW = [r + 1, r + 1, [-r, 0, -r, 0]]
    kernel_NE = [r + 1, r + 1, [-r, 0, 0, r]]
    kernel_SW = [r + 1, r + 1, [0, r, -r, 0]]
    kernel_SE = [r + 1, r + 1, [0, r, 0, r]]

    kernels = [kernel_L, kernel_R, kernel_U, kernel_D,
               kernel_NW, kernel_NE, kernel_SW, kernel_SE]

    # for padding
    m = img.shape[1] + 2 * r
    n = img.shape[2] + 2 * r
    num_channels = img.shape[0]
    ret = img.copy()
    d = np.zeros([8, m, n])  # cache of differences between output and input, 8 for 8 kernels

    for channel in range(num_channels):
        # origin = np.pad(img[:, :, channel], (r, r), mode='constant', constant_values=0)
        origin = np.pad(img[channel, :, :], (r, r), mode='edge')
        # origin = np.pad(img[channel, :, :], (r, r), mode='median')
        for it in range(iteration):
            for i, kernel in enumerate(kernels):
                cur_output = replace_mid(origin, kernel, r)
                d[i] = cur_output - origin
            origin = origin + mt.mat_absmin(d)
            ret[channel, :, :] = origin[r:-r, r:-r]

    return ret


if __name__ == '__main__':
    img_path = r"G:\学校里学的\计算可视媒体\pns_panda.jpg"
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((400, 250))
    img = np.asarray(img)  # format: rows, cols, channels
    img = np.transpose(img, (2, 0, 1))  # chw
    # output = SW_MeanFil(img, 3, 1)
    output = SW_MidFil(img, 3, 1)
    img = Image.fromarray(output)
    img.show()
    img.save(r"G:\学校里学的\计算可视媒体\pns_panda_output.jpg")
    input()
