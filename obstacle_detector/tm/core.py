import cv2
import numpy as np


def get_random_square(img, square):
    x, y, width = square
    x1, y1, x2, y2 = x, y, x + width, y + width
    return img[y1:y2, x1:x2]


def find_template(frames, coords):
    img = frames[-1]
    old_img = frames[0]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    old_img = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    template = get_random_square(old_img, coords)

    t_height, t_width = template.shape

    res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc

    bottom_right = (top_left[0] + t_width, top_left[1] + t_height)
    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    original_bottom_right = (coords[0] + coords[2], coords[1] + coords[2])
    cv2.rectangle(img, coords[:2], original_bottom_right, 255, 2)

    (x1, y1), (x2, y2) = top_left, bottom_right

    
    ravel = res.ravel()
    max_value = max(res.ravel())
    min_value = min(res.ravel())
    res = res * 255 / max_value / 4
    res = np.uint8(res)
    res = cv2.equalizeHist(res)

    return img, template, img[y1:y2, x1:x2], res
