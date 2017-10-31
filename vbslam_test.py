#!/usr/bin/python3

import numpy as np
import cv2

from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.distance_calculator import spline_dist
from obstacle_detector.points_movement import make_np_array_from_points
from obstacle_detector.points_movement import make_points_from_np_array
from obstacle_detector.tm.image_shift_calculator import find_shift_value


def create_mask_from_points_motion(
        mask, good_new, good_old,
        plane_shifting,
        M, center, height_in_pixels):

    'Now it uses heigh parameter'

    # initialize physical and computing parameters
    height, width = mask.shape[:2]
    cx, cy = center
    dx, dy, dz = plane_shifting

    # draw axis
    mask = cv2.line(mask, (cx, 0), (cx, height - 1), (255, 0, 0))
    mask = cv2.line(mask, (0, cy), (width - 1, cy), (255, 0, 0))

    # show height_in_pixels in current roi parameters

    # draw all points in green to see differents between obstacles and plane
    for new, old in zip(good_new, good_old):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.circle(mask, (a, b), 1, (0, 255, 0), -1)

    transformed_new = cv2.perspectiveTransform(np.asarray([good_new]), M)[0]
    transformed_old = cv2.perspectiveTransform(np.asarray([good_old]), M)[0]

    dists = []
    for good_new_p, transformed_new_p, transformed_old_p in zip(
            good_new, transformed_new, transformed_old):

        x, y = good_new_p.ravel()

        a, b = transformed_new_p.ravel()
        c, d = transformed_old_p.ravel()

        # filter false positive movements
        if b < d:
            continue

        dist = ((c - a) ** 2 + (b - d) ** 2 + height_in_pixels ** 2) ** 0.5
        dists.append([x, y, dist])

    dists = filter(
        lambda dist:
            (height_in_pixels ** 2 +
             abs(dx) ** 2 + 
             abs(dy) ** 2) ** 0.5 > dist[2],
        dists)
    dists = list(dists)

    if len(dists) == 0:
        return mask

    max_dist = max(dists, key=lambda dist: dist[2])[2]
    min_dist = min(dists, key=lambda dist: dist[2])[2]

    dists = map(
        lambda dist:
            [
                dist[0],
                dist[1],
                (dist[2] - min_dist) ** 0.5 * 255 / max_dist ** 0.5],
        dists)

    for i in dists:
        a, b, dist = i
        dist = int(dist)
        mask = cv2.circle(mask,(a,b),5 , (0, 0, 255),-1)

    return mask


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

    out_height, out_width, _ = cv2.pyrDown(frame[200:, 300:-300]).shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path \
            if output_video_path is not None \
            else 'output.avi',
        fourcc, 15.0, (out_width * 2, out_height))

    # initialize old_frames
    old_frame = cv2.pyrDown(cap.read()[1][200:, 300:-300])

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(old_frame)

    old_img = cv2.add(old_frame, mask)
    old_transformed_frame, pts1, M = inv_persp_new(
            cv2.pyrUp(old_img), (cx, cy),
            (roi_width, roi_length), spline_dist, 200)

    # orb initialize
    orb = cv2.ORB_create(nfeatures=250)

    # main loop #TODO fps output
    while(True):
        frame = cv2.pyrDown(cap.read()[1][200:, 300:-300])
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transformed_frame, pts1, M = inv_persp_new(
            cv2.pyrUp(frame), (cx, cy), (roi_width, roi_length), spline_dist, 200)

        kp = make_np_array_from_points(orb.detectAndCompute(old_gray,None)[0])

        new_kp, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, kp, None)

        dx, dy = find_shift_value(
            transformed_frame, old_transformed_frame, (50, 100, 150, 350))
        mask = create_mask_from_points_motion(
            mask, new_kp[st==1], kp[st==1],
            (dx, dy, 4.4),
            M, (cx // 2, cy // 2), roi_height * mask.shape[1] / roi_width)

        img = cv2.add(frame,mask)

        transformed_img, pts1, M = inv_persp_new(
            cv2.pyrUp(img), (cx, cy), (roi_width, roi_length), spline_dist, 200)

        cv2.imshow('transformed_img',transformed_img)
        cv2.imshow('out', np.concatenate((img, mask), axis=1))
        out.write(np.concatenate((img, mask), axis=1))

        old_frame = frame.copy()
        old_gray = frame_gray.copy()
        old_img = img.copy()
        old_transformed_frame = transformed_frame.copy()
        mask = np.zeros_like(old_frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('p'):
            cv2.waitKey()
        elif k == ord('s'):
            cv2.imwrite('screen.png', img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_test('../../video/6.mp4', '../results/motion_of_pointorb_out.avi')
