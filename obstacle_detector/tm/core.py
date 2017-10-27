import cv2
import numpy as np

from math import acos, pi, floor
from functools import partial

from ..utils.sum_maps_equal import sum_maps_equal
from .image_shift_calculator import find_template_in_img


def make_np_array_from_points(key_points):
    return np.asarray([[i.pt] for i in key_points], dtype='float32')


def make_points_moves_struct_from_features(pair):
    new, old = pair
    a, b = new.ravel()
    c, d = old.ravel()
    dx = c - a
    dy = d - b
    dist = (dx ** 2 + dy ** 2) ** 0.5
    cos = (a - c) / dist
    angle = acos(cos) * 180 / pi

    return (a, b), (c, d), dist, angle, (dx, dy)


def find_orb_featres(img, nfeatures=500):
    orb = cv2.ORB_create(nfeatures=500, edgeThreshold=5)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    kp, des = orb.detectAndCompute(gray, None)

    return make_np_array_from_points(kp)


def get_points_shift(img_from, img_to, features, predict_dist=15):
    if features is None or len(features) == 0:
        return []

    lk_params = dict(
        winSize  = (predict_dist, predict_dist),
        maxLevel = 5,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    features_old, st, err = cv2.calcOpticalFlowPyrLK(
        img_to, img_from, features, None, **lk_params)

    good_new = features[st==1]
    good_old = features_old[st==1]
    
    points_moves = list(map(
        make_points_moves_struct_from_features,
        zip(good_new, good_old)))

    return points_moves


def simple_shift_filter(shifts, x_range, y_range):
    result = filter(
        lambda shift:
            x_range[0] < shift[4][0] < x_range[1] and
            y_range[0] < shift[4][1] < y_range[1],
        shifts
    )

    return result


def simple_obstacles_segmentation(obstacles_map):
    obstacles_map = cv2.inRange(obstacles_map, (127, 127, 127), (255, 255, 255))
    obstacles_map = cv2.morphologyEx(obstacles_map, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8), iterations=3)
    obstacles_map = cv2.cvtColor(obstacles_map, cv2.COLOR_GRAY2BGR)
    return obstacles_map


def detect_obstacles_between_two_frames(
        current_frame_data,
        old_frame_data,
        roi,
        pre_filter=lambda img: img.frame,
        feature_detector_method=None,
        shift_detection_method=None,
        shift_filter_method=None,
        obstacles_segmentation_method=None,
        obstacles_filter_method=None):

    "TODO docs"

    # intialization block TODO
    if feature_detector_method is None:
        feature_detector_method = find_orb_featres

    if shift_detection_method is None:
        shift_detection_method = get_points_shift

    if shift_filter_method is None:
        shift_filter_method = simple_shift_filter

    if obstacles_segmentation_method is None:
        obstacles_segmentation_method = simple_obstacles_segmentation

    if obstacles_filter_method is None:
        pass

    current_frame = current_frame_data.frame
    old_frame = old_frame_data.frame
    x1, y1, x2, y2 = roi


    def obstacle_map_between_two_frames(current_frame, old_frame, div=15):
        current_frame = current_frame[y1:y2, x1:x2].copy()

        old_x1, old_y1, old_x2, old_y2 = find_template_in_img(
            current_frame, old_frame)
        old_frame = old_frame[old_y1:old_y2, old_x1:old_x2]

        features = feature_detector_method(pre_filter(old_frame))
        features_shifts = shift_detection_method(
            old_frame, current_frame, features)
        features_shifts = shift_filter_method(
            features_shifts, (-3, 3), (-10, -2))

        test = np.zeros_like(current_frame)

        for shift in features_shifts:
            test = cv2.line(
                test, shift[0], shift[1], (255, 255, 255), 1)
            current_frame = cv2.line(
                current_frame, shift[0], shift[1], (255, 255, 255), 1)
        return test, current_frame


    obstacles_map, obstacles_on_frame = obstacle_map_between_two_frames(
        current_frame, old_frame, 1)
    obstacles_map = obstacles_segmentation_method(obstacles_map)

    return obstacles_map


def detect_obstacles(
        frames,
        **kwargs):

    'TODO DOCS'

    partial_detector =  partial(
        detect_obstacles_between_two_frames,
        **kwargs)

    current_frame_data = frames[-1]
    old_frame_data = frames[0]

    obstacles_map = sum_maps_equal([
        partial_detector(current_frame_data, frames[-i])
        for i in range(14, len(frames), 1)])
    obstacles_map = cv2.inRange(
        obstacles_map, (128, 128, 128), (255, 255, 255))

    obstacles_map = cv2.cvtColor(obstacles_map, cv2.COLOR_GRAY2BGR)
#    obstacles_on_frame *= 255
    x1, y1, x2, y2 = kwargs['roi']
    obstacles_on_frame = cv2.bitwise_or(
        current_frame_data.frame[y1:y2, x1:x2], obstacles_map)

    return obstacles_map, obstacles_on_frame
