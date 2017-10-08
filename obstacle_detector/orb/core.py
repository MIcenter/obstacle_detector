import cv2
import numpy as np

import itertools
from math import floor, log2, pi, acos
from functools import lru_cache


def make_np_array_from_points(key_points):
    return np.asarray([[i.pt] for i in key_points], dtype='float32')


def make_points_moves_struct_from_features(pair):
    new, old = pair
    a, b = new.ravel()
    c, d = old.ravel()
    dist = ((c - a) ** 2 + (d - b) ** 2)
    cos = (a - c) / (dist ** 0.5)
    angle = acos(cos) * 180 / pi

    return (a, b), (c, d), dist, angle


def find_features(img):
    orb = cv2.ORB_create(
        nfeatures=200, edgeThreshold=5, scaleFactor=2, patchSize=62)

    kp = orb.detect(img, None)
    kp, _ = orb.compute(img, kp)
    points = make_np_array_from_points(orb.detect(img, None))
    return points


def normalize(points_moves):
    max_dist = max(points_moves, key=lambda item: item[2])[2]
    min_dist = min(points_moves, key=lambda item: item[2])[2]
    mean_dist = \
        sum(map(lambda item: item[2], points_moves)) / len(points_moves)

    # HOT FIX
    max_dist = 10
    points_moves = map(
        lambda item: (
            item[0], item[1], item[2] * 255 // max_dist, item[3]),
        points_moves)

    return points_moves

def filter_points(points_moves):
    points_moves = normalize(points_moves)

    points_moves = filter(
        lambda item:
            item[2] > 0,
            # item[2] > min_dist * 2,
        points_moves)

    return points_moves


def get_points_moves(points, img_gray, old_gray, predict_dist=15):
    lk_params = dict(
        winSize  = (predict_dist, predict_dist),
        maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # calculate optical flow
    points_old, st, err = cv2.calcOpticalFlowPyrLK(
        img_gray, old_gray, points, None, **lk_params)
    #Select good points
    good_new = points[st==1]
    good_old = points_old[st==1]
    # draw the tracks
    

    points_moves = list(map(
        make_points_moves_struct_from_features,
        zip(good_new, good_old)))

    return points_moves


def construct_moving_vectors():
    pass


def draw_points_moves(img, points_moves):
    angle_map = np.zeros_like(img)
    angle_map[:,:,:] = (0, 127, 0)

    dist_map = np.zeros_like(img)

    for i, ((x, y), (x_old, y_old), dist, angle) in enumerate(points_moves):
        # if dist < 50:
        #    continue
        dist_map = cv2.line(
            dist_map, (x,y), (x_old, y_old),
            (dist, dist, dist), int(log2(dist // 16 + 1)))

        angle = int(angle)
        angle_map = cv2.circle(
            angle_map, (int(x), int(y)), 4, (angle, dist, 255), -1)

    angle_map = cv2.cvtColor(angle_map, cv2.COLOR_HSV2BGR)
    result_img = img.copy()
    mask = cv2.inRange(angle_map, (1, 1, 1), (255, 255, 255))
    mask = cv2.bitwise_not(mask)
    result_img = cv2.bitwise_and(result_img, result_img, mask=mask)
    result_img = cv2.add(result_img, angle_map)

    return result_img, angle_map, dist_map


def handle_img(img, old_images):
    old_img = old_images[-2]
    too_old_img = old_images[-len(old_images)]

    img = cv2.blur(img, (5, 5))
    old_img = cv2.blur(old_img, (5, 5))
    too_ld_img = cv2.blur(too_old_img, (5, 5))

    points = find_features(img)

    if points is None or len(points) == 0:
        return img, None, None, None

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    old_gray = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
    too_old_gray = cv2.cvtColor(too_old_img, cv2.COLOR_BGR2GRAY)

    points_moves = list(get_points_moves(points, img_gray, old_gray))
    too_old_points_moves = list(
        get_points_moves(points, img_gray, too_old_gray))

    if len(points_moves) == 0 or len(too_old_points_moves) == 0:
        return img, None, None, None

    points_diff = list(
        map(lambda pts: 
            (
                pts[0][0],
                pts[0][1],
                pts[1][2] / pts[0][2],
                abs(pts[0][3] - pts[1][3])), zip(points_moves, too_old_points_moves)))
    points_diff = list(
        filter(
            lambda pt:
                (pt[3] < 0.25) and (2 < pt[2] < 5), points_diff))

    if len(points_diff) == 0:
        return img, None, None, None
    points_diff = filter_points(points_diff)

    result_img, angle_map, dist_map = draw_points_moves(img, points_diff)

    return img, angle_map, dist_map, result_img
