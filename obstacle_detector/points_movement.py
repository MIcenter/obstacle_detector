import cv2
import numpy as np

from math import acos, pi


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
        nfeatures=200, edgeThreshold=25, scaleFactor=2, patchSize=62)

    kp = orb.detect(img, None)
    kp, _ = orb.compute(img, kp)
    points = make_np_array_from_points(orb.detect(img, None))
    return points


def get_points_moves(points, img_gray, old_gray, predict_dist=15):
    if points is None or len(points) == 0:
        return []

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
