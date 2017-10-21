#!/usr/bin/python3

import numpy as np
import cv2

from collections import deque

import obstacle_detector.back_sub as back_sub
import obstacle_detector.tm as tm
from obstacle_detector.frame_queue import Frame_queue

from obstacle_detector.distance_calculator import spline_dist
from obstacle_detector.perspective import find_center_point
from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.perspective import regress_perspecive

from obstacle_detector.utils.gabor_filter import gabor_filter

def video_test(input_video_path=None, output_video_path=None):
    cx = 595
    cy = 303
    roi_width = 25
    roi_length = 90

    cap = cv2.VideoCapture(
        input_video_path \
            if input_video_path is not None \
            else input('enter video path: '))

    transformed_frames = Frame_queue()
    original_frames = deque()

    ret, frame = cap.read()
    for i in range(15):
        original_frames.append(frame)

        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)
        transformed_frames.append(img)

        ret, frame = cap.read()

    cx, cy = find_center_point(original_frames, (400, 100, 800, 719))

    height, width, _ = frame.shape
    out_height, out_width, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path \
            if output_video_path is not None \
            else 'output.avi',
        fourcc, 15.0, (out_width * 4, out_height))

    shift = 23

    while(ret):
        ret, frame = cap.read()

        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)

        transformed_frames.popleft()
        transformed_frames.append(img)

        # img_gray, old_img_gray, match, absdiff = tm.find_obstacles(transformed_frames, (0, 50, 200, 400), method='sparse')

        obstacles_map, obstacles_on_frame = tm.detect_obstacles(
            transformed_frames, (0, 250, 200, 550), gabor_filter)

        obstacles_map2, obstacles_on_frame2 = tm.detect_obstacles(
            transformed_frames, (0, 250, 200, 550), gabor_filter)

        cv2.imshow('obstacles', obstacles_map)
        cv2.imshow('obstacles on frame', obstacles_on_frame)
        cv2.imshow('original', transformed_frames[-1].frame)

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

video_test('../../video/6.mp4', '../results/back_sub_out.avi')
