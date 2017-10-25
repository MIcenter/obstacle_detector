#!/usr/bin/python3

import numpy as np
import cv2

from collections import deque

from obstacle_detector.distance_calculator import spline_dist
from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.perspective import regress_perspecive

from obstacle_detector.depth_mapper import calculate_depth_map

def video_test(input_video_path=None, output_video_path=None):
    cx = 595
    cy = 303
    roi_width = 25
    roi_length = 90

    cap = cv2.VideoCapture(
        input_video_path \
            if input_video_path is not None \
            else input('enter video path: '))

    old_images = deque()
    original_frames = deque()

    ret, frame = cap.read()
    for i in range(15):
        original_frames.append(frame)

        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)
        old_images.append(img)

        ret, frame = cap.read()

    height, width, _ = frame.shape
    out_height, out_width, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path \
            if output_video_path is not None \
            else 'output.avi',
        fourcc, 15.0, (out_width * 4, out_height))

    left = cv2.imread('aloeL.jpg')
    right = cv2.imread('aloeR.jpg')
    while(ret):
        ret, frame = cap.read()

        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)
        old_images.popleft()
        old_images.append(img)

        dm = calculate_depth_map(left, right)
        dm = cv2.equalizeHist(dm)
        cv2.imshow('dm', dm)

        dm = cv2.cvtColor(dm, cv2.COLOR_GRAY2BGR)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('screen.png', img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_test('../../video/6.mp4', '../results/depth_map_out.avi')
