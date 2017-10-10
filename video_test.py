#!/usr/bin/python3

import numpy as np
import cv2

from collections import deque

import obstacle_detector.back_sub as back_sub

from obstacle_detector.distance_calculator import spline_dist
from obstacle_detector.perspective import find_center_point
from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.orb import handle_img


def video_test(input_video_path=None, output_video_path=None):
    cx = 595 #new calibration
    cy = 303
    roi_width = 25
    roi_length = 90

    cap = cv2.VideoCapture(
        input_video_path \
            if input_video_path is not None \
            else input('enter video path: '))

    old_images = deque()

    ret, frame = cap.read()

    original_frames = deque()

    for i in range(15):
        original_frames.append(frame)
        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)
        # old_img = cv2.blur(img, (3, 3))
        img = cv2.blur(img, (7, 7))
        old_images.append(img)
        ret, frame = cap.read()

    cy, cy = find_center_point(original_frames, (400, 100, 800, 719))

    height, width, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path \
            if output_video_path is not None \
            else 'output.avi',
        fourcc, 15.0, (width * 4, height))

    frame_number = 0
    shift = 33

    while(ret):
        ret, frame = cap.read()

        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)

        # img, angle_map, dist_map, result_img = handle_img(img, old_images)

        old_images.popleft()
        img = cv2.blur(img, (7, 7))
        old_images.append(img)

        new, old, sub_img = back_sub.calc_diff(
            old_images, shift_per_frame=shift)
#        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
#        sub_img = cv2.equalizeHist(sub_img)
#        sub_img = cv2.cvtColor(sub_img, cv2.COLOR_GRAY2BGR)

        #if img is None or angle_map is None or dist_map is None:
        #    continue

        cv2.imshow(
            'img',
            np.concatenate((img, new, old, sub_img), axis=1))
        # cv2.imshow('frame', frame)
        out.write(np.concatenate((img, new, old, sub_img), axis=1))


        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('m'):
            shift += 1
            print(shift)
        elif k == ord('l'):
            shift -= 1
            print(shift)
        elif k == ord('s'):
            cv2.imwrite('screen.png', img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_test('../../video/1.mp4', '../results/output.avi')

