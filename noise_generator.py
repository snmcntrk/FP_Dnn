import numpy as np
import math


def noise_generator(noise_type, image):
    if len(image.shape) == 3:
        row, col, ch = image.shape
    else:
        row, col = image.shape
    if noise_type == "gauss":
        mean = 0
        var = 1.5
        sigma = var ** 0.5
        gauss = np.array(image.shape)
        if len(image.shape) == 3:
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
        else:
            gauss = np.random.normal(mean, sigma, (row, col))
            gauss = gauss.reshape(row, col)

        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "saltNpepper":
        s_vs_p = 0.5
        amount = 0.05
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        out[tuple(coords)] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0
        return out
    elif noise_type == "concave":
        # Concave effect
        noisy = np.zeros(image.shape, dtype=image.dtype)
        for i in range(row):
            for j in range(col):
                offset_x = int(30.0 * math.sin(2 * 3.14 * i / (2 * col)))
                if j + offset_x < col-2:
                    noisy[i, j] = image[i, (j + offset_x) % col]
                else:
                    noisy[i, j] = 255
        return noisy

    elif noise_type == "both":
        # Both horizontal and vertical
        img_output = np.zeros(image.shape, dtype=image.dtype)

        for i in range(row):
            for j in range(col):
                offset_x = int(5.0 * math.sin(2 * 3.14 * i / 150))
                offset_y = int(5.0 * math.cos(2 * 3.14 * j / 150))
                if i + offset_y < row-2 and j + offset_x < col-2:
                    img_output[i, j] = image[(i + offset_y) % row, (j + offset_x) % col]
                else:
                    img_output[i, j] = 255
        return img_output
    else:
        return image


