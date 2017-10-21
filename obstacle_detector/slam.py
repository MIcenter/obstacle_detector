import cv2
import numpy as np


def get_rectangle_from_img(img, rectangle):
    x, y, width, height = rectangle
    x1, y1, x2, y2 = x, y, x + width, y + height
    return img[y1:y2, x1:x2].copy()


def find_shift_value(img, old_img, coords=(0, 50, 300, 200)):
    old_template = get_rectangle_from_img(old_img, coords)

    res = cv2.matchTemplate(img, old_template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc

    t_height, t_width = old_template.shape[:2]

    return top_left[1] + t_height - coords[1] - coords[3]


def stich_two_images(slam_map, img1, img2, shift_y=None):
    print(find_shift_value(img2, img1))
    if shift_y is None:
        shift_y = find_shift_value(img1, img2)
        shift_y = min(shift_y, 0)

    shifted_img2 = np.roll(img2, shift_y, axis=0)
    shifted_img2 = shifted_img2[:-shift_y,:]
    return np.concatenate((shifted_img2, slam_map))
