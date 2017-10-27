import cv2
import numpy as np

from .tm.image_shift_detector import find_shift_value


def stich_two_images(
        slam_map, img1, img2,
        x_excess=20,
        shift_x=None, shift_y=None):

    if shift_y is None:
        y_from = img2.shape[0] - 200
        y_to = img2.shape[0] - 100
        shift_x, shift_y = find_shift_value(
            img1,
            img2,
            coords=(x_excess, y_from, -x_excess, y_to))
        shift_y = min(shift_y, 0) // 8

    shifted_img2 = np.roll(img2, shift_y, axis=0)
    shifted_img2 = np.roll(shifted_img2, shift_x, axis=1)
    shifted_img2 = shifted_img2[:-shift_y,x_excess:-x_excess]
    return np.concatenate((shifted_img2, slam_map)), shift_x, shift_y
