#!/usr/bin/python3

import numpy as np
import cv2

from obstacle_detector.utils.gabor_filter import gabor_filter
from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.distance_calculator import spline_dist
from obstacle_detector.perspective import find_center_point


def video_test(input_video_path=None, output_video_path=None):
    cx = 595
    cy = 303
    roi_width = 25
    roi_length = 90

    cap = cv2.VideoCapture(
        input_video_path \
            if input_video_path is not None \
            else input('enter video path: '))

    original_frames = []

    ret, frame = cap.read()
    for i in range(15):
        original_frames.append(frame)
        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)
        ret, frame = cap.read()

    cy, cy = find_center_point(original_frames, (400, 100, 800, 719))

    while(ret):
        ret, frame = cap.read()

        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)
        img = gabor_filter(img)

        canny = cv2.Canny(img, 50, 200)

        cv2.imshow('original image', img)
        cv2.imshow('after canny', canny)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

video_test('../../video/6.mp4', '../results/back_sub_out.avi')
