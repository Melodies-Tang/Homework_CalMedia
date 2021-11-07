import os
from tqdm import tqdm
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
    m = img.shape[1] + 2 * r
    n = img.shape[2] + 2 * r
    num_channels = img.shape[0]

    ret = img.copy()
    # for it in range(iteration):
    #     for channel in range(num_channels):
    #         origin = np.pad(ret[channel, :, :], (r, r), mode='constant', constant_values=0)  # padding by zero
    #         for row in range(img.shape[1]):
    #             for col in range(img.shape[2]):
    #                 ret[row][col] = convolve(kernels, r, row, col, origin)

    d = np.zeros([8, m, n])  # cache of differences between output and input, 8 for 8 kernels

    print("Number of processing bar based on specific algorithm")
    with tqdm(total=8 * iteration * num_channels) as pbar:
        pbar.set_description("Processing")
        for channel in range(num_channels):
            # padding by values at edge; which let pixels near edge influenced much more by global image edge
            # origin = np.pad(img[:, :, channel], (r, r), mode='edge')
            origin = np.pad(ret[channel, :, :], (r, r), mode='constant', constant_values=0)  # padding by zero
            for it in range(iteration):
                for i, kernel in enumerate(kernels):
                    # learned some convolution operation for speedup
                    cur_output = sci.correlate2d(origin, kernel, 'same')
                    d[i] = cur_output - origin
                    pbar.update(1)
                origin = origin + mt.mat_absmin(d)
                ret[channel, :, :] = origin[r:-r, r:-r]
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
    result = np.pad(result, (r, r), 'edge')
    return result


def replace_mid(origin, kernel, r):
    padm = origin.shape[0]
    padn = origin.shape[1]
    retm = padm - 2 * r
    retn = padn - 2 * r
    ret = np.zeros((retm, retn))
    for row in range(r, retm + r):
        for col in range(r, retn + r):
            tmp = origin[row + kernel[2][0]:row + kernel[2][1], col + kernel[2][2]:col + kernel[2][3]]
            ret[row - r][col - r] = np.median(tmp)
    return ret


# Not suitable if there is an area with too many noisy pixels (1)
# or noise like Gaussian noise whose value distributed evenly
# Larger radius can help, but cause color leakage meanwhile
# Better for pepper-and-salt noise that only take 0 or 255

def SW_MidFil(img, r, iteration=1):
    # due to difference between mid and mean filter, size of kernels for median filter are different
    kernel_L = [2 * r + 1, r + 1, [-r, r, -r, 0]]  # [rows, cols, [r0, r1, c0, c1]]
    kernel_R = [2 * r + 1, r + 1, [-r, r, 0, r]]
    kernel_U = [r + 1, 2 * r + 1, [-r, 0, -r, r]]
    kernel_D = [r + 1, 2 * r + 1, [0, r, -r, r]]
    kernel_NW = [r + 1, r + 1, [-r, 0, -r, 0]]
    kernel_NE = [r + 1, r + 1, [-r, 0, 0, r]]
    kernel_SW = [r + 1, r + 1, [0, r, -r, 0]]
    kernel_SE = [r + 1, r + 1, [0, r, 0, r]]

    kernels = [kernel_L, kernel_R, kernel_U, kernel_D,
               kernel_NW, kernel_NE, kernel_SW, kernel_SE]

    ret = img.copy()
    # for padding
    m = ret.shape[1]
    n = ret.shape[2]
    num_channels = ret.shape[0]
    d = np.zeros([8, m, n])  # cache of differences between output and input, 8 for 8 kernels

    print("Number of processing bar based on specific algorithm")
    with tqdm(total=8 * num_channels * iteration) as pbar:
        pbar.set_description("Processing")
        for it in range(iteration):
            for channel in range(num_channels):
                # pad = np.pad(img[channel, :, :], (r, r), mode='constant', constant_values=0)
                pad = np.pad(ret[channel, :, :], (r, r), mode='edge')
                # pad = np.pad(img[channel, :, :], (r, r), mode='median')
                for i, kernel in enumerate(kernels):
                    cur_output = replace_mid(pad, kernel, r)
                    d[i] = cur_output - ret[channel, :, :]
                    pbar.update(1)
                ret[channel, :, :] = ret[channel, :, :] + mt.mat_absmin(d)
    return ret


if __name__ == '__main__':
    # img_path = "/home/melodies/Downloads/Lenna.jpg"
    img_path = input("Please provide the full path of input image (including file extension): ")
    while not os.path.isfile(img_path):
        img_path = input("Path not exist, please check again: ")

    filter_type = (input("Please choose the operation you want:\ns: smoothing d: denoising\n(Can do better on "
                         "pepper-and-salt noise)\n")).lower()[0]
    while filter_type != "s" and filter_type != "d":
        filter_type = input("Please type correct instruction: s for smoothing, d for denoising: ")

    radius = 3
    iteration = 1
    customize = (input("You can type 'y' to customize parameter of filter radius and iteration number\nOr type any other key to use default parameter: radius = 3 and iteration = 1\n").lower())
    if customize == "y":
        print("Both parameter should be an integer")
        radius = int(input("Your filter radius (larger the radius, coarser the image): "))
        iteration = int(input("How many times the filtering you want: "))

    img = Image.open(img_path)
    img = np.asarray(img)  # format: rows, cols, channels
    img = np.transpose(img, (2, 0, 1))  # chw. transpose for better debugging

    if filter_type == 's':
        print("An edge-preserving smoothing will take place")
        output = SW_MeanFil(img, radius, iteration)
    else:
        print("An edge-preserving denoising will take place")
        output = SW_MidFil(img, radius, iteration)

    img = np.transpose(img, (1, 2, 0))  # hwc
    output = np.transpose(output, (1, 2, 0))  # hwc
    img = np.hstack((img, output))
    img = Image.fromarray(img)
    img.show()

    output = Image.fromarray(output)
    full_name = os.path.splitext(img_path)
    save_path = full_name[0] + "_SW" + ("smoothing" if filter_type == 's' else "denoising") + full_name[-1]
    output.save(save_path)
    print("Output image saved to " + save_path)
