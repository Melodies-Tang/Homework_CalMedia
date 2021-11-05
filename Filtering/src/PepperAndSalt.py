from PIL import Image
import numpy as np
import random


def generate_pns_noise(img, ratio=0.3):
    num_channels = img.shape[0]
    ret = img.copy()
    for i in range(ret.shape[1]):
        for j in range(ret.shape[2]):
            r = random.random()
            if r < ratio:
                r = random.random()
                black_or_white = 0 if r < 0.5 else 255
                for channel in range(num_channels):
                    ret[channel][i][j] = black_or_white
    return ret


if __name__ == '__main__':
    img_path = "/home/melodies/Downloads/Lenna.jpg"
    img = Image.open(img_path)
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # chw. transpose for better debugging
    output = generate_pns_noise(img)
    output = np.transpose(output, (1, 2, 0))  # hwc
    img = Image.fromarray(output)
    img.show()
    img.save("/home/melodies/Downloads/test_pns.png")
