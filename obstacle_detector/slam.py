import cv2
import numpy as np


def get_rectangle_from_img(img, rectangle):
    x1, y1, x2, y2 = rectangle
    return img[y1:y2, x1:x2].copy()


def find_shift_value(img, old_img, coords):
    old_template = get_rectangle_from_img(old_img, coords)

    res = cv2.matchTemplate(img, old_template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    print('top_left', top_left)

    t_height, t_width = old_template.shape[:2]

    return top_left[0] - coords[0], top_left[1] - coords[1]


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
