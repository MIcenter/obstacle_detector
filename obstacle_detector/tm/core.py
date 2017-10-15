import cv2
import numpy as np


def find_matches(gray1, gray2, shift_distance):
    orb = cv2.ORB_create()

    gray1 = cv2.bilateralFilter(gray1, 9, 75, 75)
    gray2 = cv2.bilateralFilter(gray2, 9, 75, 75)

    gray1 - cv2.equalizeHist(gray1)
    gray2 - cv2.equalizeHist(gray2)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        return np.concatenate((gray1, gray2), axis=1)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING,  crossCheck=True)

    matches = bf.match(des1, des2)

    matches = filter(lambda x: x.distance > shift_distance, matches)
    matches = sorted(matches, key = lambda x: x.distance)

    gray3 = cv2.drawMatches(gray1, kp1, gray2, kp2, matches[:], None, flags=2)
    return gray3


def get_random_square(img, square):
    x, y, width = square
    x1, y1, x2, y2 = x, y, x + width, y + width
    return img[y1:y2, x1:x2].copy()


def find_template(frames, coords):
    img = frames[-1].copy()
    old_img = frames[-3].copy()

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
    
    shift_distance = y1 - coords[1]
    matches = find_matches(match, old_template, shift_distance)

    return img, old_img, matches, absdiff
