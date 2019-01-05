import numpy as np
from scipy.misc import imread, imresize, imsave
import torch

# debug
# import matplotlib.pyplot as plt

def load_image(filepath):
    image = imread(filepath)
    # debug
    # plt.imshow(image)
    # plt.show()
    if len(image.shape) < 3:
        image = np.expand_dims(image, axis=2)
        image = np.repeat(image, 3, axis=2)
    image = np.transpose(image, (2, 0, 1)) # (channels, width, height)
    image = torch.from_numpy(image)
    # print(image)
    image = torch.FloatTensor(image.size()).copy_(image)
    temp_min = image.min()
    temp_max = image.max()
    # print("min %f, max %f" % (temp_min, temp_max))
    image.add_(-temp_min).mul_(1.0 / (temp_max - temp_min))
    image = image.mul_(2).add_(-1)
    # print(image)
    return image


def save_image(image, filename):
    image = image.add_(1).div_(2)
    image = image.numpy()
    image *= 255.0
    image = image.clip(0, 255)
    image = np.transpose(image, (1, 2, 0))
    image = image.astype(np.uint8)
    imsave(filename, image)
    print ("Image saved as {}".format(filename))

def is_image(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg"])
