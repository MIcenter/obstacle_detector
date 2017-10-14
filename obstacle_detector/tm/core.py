import cv2
import numpy as np


def get_random_square(img, square):
    x, y, width = square
    x1, y1, x2, y2 = x, y, x + width, y + width
    return img[y1:y2, x1:x2].copy()


def find_template(frames, coords):
    img = frames[-1].copy()
    old_img = frames[0].copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    old_img = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)

    old_template = get_random_square(old_img, coords)

    res = cv2.matchTemplate(img, old_template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc

    t_height, t_width = old_template.shape

    bottom_right = (top_left[0] + t_width, top_left[1] + t_height)
    original_bottom_right = (coords[0] + coords[2], coords[1] + coords[2])
    (x1, y1), (x2, y2) = top_left, bottom_right

    match = img[y1:y2, x1:x2].copy()

    absdiff = cv2.absdiff(old_template, match)
    absdiff = cv2.equalizeHist(absdiff)

    img = cv2.rectangle(img, coords[:2], original_bottom_right, 255, 1)
    old_img = cv2.rectangle(old_img, top_left, bottom_right, 255, 2)
    
    ravel = res.reshape((-1))
    max_value = max(ravel)
    min_value = min(ravel)
    res = res * 255 / max_value / 4
    res = np.uint8(res)
    res = cv2.equalizeHist(res)

    return img, old_img, match, old_template, absdiff
