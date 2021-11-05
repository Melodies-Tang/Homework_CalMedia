from PIL import Image
import numpy as np


def generate_gaussian_noise(img, mean=0, var=0.1):
    num_channels = img.shape[0]
    ret = img.copy()
    for channel in range(num_channels):
        cur = np.array(ret[channel]/255.0, dtype=float)
        noise = np.random.normal(mean, var, size=cur.shape)
        cur += noise
        cur = np.clip(cur, 0, 1)
        cur = np.uint8(cur * 255)
        ret[channel] = cur
    return ret


if __name__ == '__main__':
    img_path = "/home/melodies/Downloads/Lenna.jpg"
    img = Image.open(img_path)
    img = np.asarray(img)
    img = np.transpose(img, (2, 0, 1))  # chw. transpose for better debugging
    output = generate_gaussian_noise(img)
    output = np.transpose(output, (1, 2, 0))  # hwc
    img = Image.fromarray(output)
    img.show()
    img.save("/home/melodies/Downloads/test_gauss.png")
