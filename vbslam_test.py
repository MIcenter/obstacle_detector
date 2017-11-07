#!/usr/bin/python3

import numpy as np
import cv2

from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.distance_calculator import spline_dist
from obstacle_detector.points_movement import make_np_array_from_points
from obstacle_detector.points_movement import make_points_from_np_array
from obstacle_detector.tm.image_shift_calculator import find_shift_value

from obstacle_detector.utils.keyboard_interface import handle_keyboard
from obstacle_detector.utils.gabor_filter import gabor_filter
from obstacle_detector.utils.sum_maps_equal import sum_maps_equal


def unite_points(img, iterations):
    for i in range(1 + iterations):
        for it in range(i):
            img = cv2.pyrDown(img)
        for it in range(i):
            img = cv2.pyrUp(img)
    return img


def create_point_shift_structure(
        good_new, good_old, M):

    """TODO doc"""

    transformed_new = cv2.perspectiveTransform(np.asarray([good_new]), M)[0]
    transformed_old = cv2.perspectiveTransform(np.asarray([good_old]), M)[0]

    dists = []

    for good_new_p, good_old_p, transformed_new_p, transformed_old_p in zip(
            good_new, good_old, transformed_new, transformed_old):

        x, y = good_new_p.ravel()
        x_old, y_old = good_old_p.ravel()

        a, b = transformed_new_p.ravel()
        c, d = transformed_old_p.ravel()

        dists.append(((x, y), (x_old, y_old), (a, b, c, d)))

    return dists


def find_homography_anomalies(shifted_points):

    """Find points in shifted points where x coord has been changed"""

    x_shifted_points = []

    for pt in shifted_points:
        (x, y), (x_old, y_old), (a, b, c, d) = pt

        if abs(c-a) > 2:
            radius = max(1, ((x - x_old) ** 2 + (y - y_old) ** 2) ** 0.5)
            x_shifted_points.append((x, y, radius))

    return x_shifted_points


def create_mask_from_points_motion(
        mask, good_new, good_old,
        M, center):

    'Now it uses few different and simple but effective methods'

    # initialize physical and computing parameters
    cx, cy = center

    shifted_points = create_point_shift_structure(good_new, good_old, M)
    x_shifted_points = find_homography_anomalies(shifted_points)
    x_shifted_points = list(filter(
        lambda pt:
            pt[1] > cy and pt[2] < 10,
        x_shifted_points))

    little_shifts = filter(
        lambda pt:
            1 <= pt[2] <= 1,
        x_shifted_points)

    middle_shifts = filter(
        lambda pt:
            2 <= pt[2] <= 4,
        x_shifted_points)

    big_shifts = filter(
        lambda pt:
            3 <= pt[2] <= 5,
        x_shifted_points)

    large_shifts = filter(
        lambda pt:
            4 <= pt[2] <= 9,
        x_shifted_points)

    shifts = [
        little_shifts,
        middle_shifts,
        big_shifts,
        large_shifts]

    masks = []

    for points in shifts:
        new_mask = np.zeros_like(mask)
        for pt in points:
            x, y, radius = pt
            new_mask = cv2.circle(new_mask,(x,y), int(radius), (255, 0, 0),-1)

        masks.append(new_mask)

    return masks, x_shifted_points


def get_obstacles_map(
        mask, new_kp, kp, M, center):

    """TODO DOCS"""

    masks, x_shifted = create_mask_from_points_motion(
        mask, new_kp, kp,
        M, center)

    drawed_contours_list = []
    obstacles_blocks_list = []

    for mask in masks:
        mask = cv2.pyrDown(mask.copy())

        opening = cv2.morphologyEx(
            mask[..., 0], cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8), iterations=3)

        opening = cv2.inRange(opening, 1, 255)

        drawed_contours = np.zeros_like(mask)
        obstacles_blocks = np.zeros_like(mask)

        ret, thresh = cv2.threshold(opening, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        drawed_contours = cv2.drawContours(
            drawed_contours, contours, -1, (255,0, 255), 1)

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            obstacles_blocks = cv2.rectangle(
                obstacles_blocks, (x, y), (x + w, y + h),
                (0, 0, 255), thickness=cv2.FILLED)
            drawed_contours = cv2.rectangle(
                drawed_contours, (x,y),(x+w,y+h),(0,0,255),1)

        drawed_contours_list.append(drawed_contours)
        obstacles_blocks_list.append(obstacles_blocks)

    return masks, drawed_contours_list, obstacles_blocks_list


def video_test(input_video_path=None, output_video_path=None):
    # physical and computing parameters
    cx = 595 - 300
    cy = 303 - 200

    roi_width = 6
    roi_length = 20
    roi_height = 4.4

    # video output block
    cap = cv2.VideoCapture(
        input_video_path \
            if input_video_path is not None \
            else input('enter video path: '))
    ret, frame = cap.read()

    out_height, out_width, _ = frame[200:, 300:-300].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path \
            if output_video_path is not None \
            else 'output.avi',
        fourcc, 3.0, (out_width, out_height))

    # initialize old_frames
    old_frame = cap.read()[1][200:, 300:-300]

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(old_frame)

    old_img = cv2.add(old_frame, mask)
    old_transformed_frame, pts1, M = inv_persp_new(
            old_img, (cx, cy),
            (roi_width, roi_length), spline_dist, 200)

    # orb initialize
    orb = cv2.ORB_create(nfeatures=3500)

    # main loop #TODO fps output
    frame = old_frame.copy()
    while(True):
        for i in range(4):
            frame = cap.read()[1][200:, 300:-300]

        #frame = cv2.pyrUp(cv2.pyrDown(frame))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transformed_frame, pts1, M = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)

        kp = make_np_array_from_points(orb.detectAndCompute(old_gray,None)[0])

        new_kp, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, kp, None)

        dx, dy = find_shift_value(
            transformed_frame, old_transformed_frame, (50, 300, 250, 550))

        masks, drawed_contours, obstacles_blocks_list = get_obstacles_map(
            mask, new_kp[st==1], kp[st==1], M, (cx, cy))

        img = frame.copy()
        obstacles = sum_maps_equal(obstacles_blocks_list, [1, 2, 4, 8])#, [0.1, 0.2, 0.3, 0.4])

        cv2.imshow(
            'obstacles',
            cv2.add(
                frame, cv2.pyrUp(obstacles)))
        cv2.imshow('img', cv2.addWeighted(
            img, 0.5,
            cv2.pyrUp(drawed_contours[2]), 0.5,
            0))
        out.write(cv2.addWeighted(
            img, 0.5,
            cv2.pyrUp(drawed_contours[2]), 0.5,
            0))

        old_frame = frame.copy()
        old_gray = frame_gray.copy()
        old_img = img.copy()
        old_transformed_frame = transformed_frame.copy()
        mask = np.zeros_like(old_frame)

        if handle_keyboard(screenshot_image=None) == 1:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_test('../../video/2.mp4', '../results/motion_of_pointorb_out_2_united.avi')
