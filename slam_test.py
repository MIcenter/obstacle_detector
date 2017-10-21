#!/usr/bin/python3

import numpy as np
import cv2

from collections import deque

import obstacle_detector.tm as tm
from obstacle_detector.frame_queue import Frame_queue
from obstacle_detector.slam import *

from obstacle_detector.distance_calculator import spline_dist
from obstacle_detector.perspective import find_center_point
from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.perspective import regress_perspecive


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
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 100)
        transformed_frames.append(img)

        ret, frame = cap.read()

    cy, cy = find_center_point(original_frames, (400, 100, 800, 719))

    past_img = transformed_frames[-5].frame
    while(ret):
        ret, frame = cap.read()

        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 100)
        #img_gray, old_img_gray, match, absdiff = tm.find_obstacles(transformed_frames, (0, 50, 200, 400), method='sparse')

        transformed_frames.popleft()
        transformed_frames.append(img)

        cv2.imshow('current', transformed_frames[-1].frame)
        cv2.imshow('past', transformed_frames[0].frame)

        past_img = stich_two_images(
            past_img,
            transformed_frames[-3].frame,
            transformed_frames[-1].frame)

        cv2.imshow('slam', past_img)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cv2.imwrite('slam.jpg', past_img)
    cap.release()
    cv2.destroyAllWindows()

video_test('../../video/1.mp4', '../results/back_sub_out.avi')
