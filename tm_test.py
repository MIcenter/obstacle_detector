#!/usr/bin/python3

import numpy as np
import cv2

from collections import deque

import obstacle_detector.back_sub as back_sub
import obstacle_detector.tm as tm

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

    old_images = deque()
    original_frames = deque()

    ret, frame = cap.read()
    for i in range(15):
        original_frames.append(frame)

        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)
        img = cv2.blur(img, (3, 3))
        old_images.append(img)

        ret, frame = cap.read()

    cy, cy = find_center_point(original_frames, (400, 100, 800, 719))

    height, width, _ = frame.shape
    out_height, out_width, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path \
            if output_video_path is not None \
            else 'output.avi',
        fourcc, 15.0, (out_width * 4, out_height))

    frame_number = 0
    shift = 23

    while(ret):
        ret, frame = cap.read()

        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)

        old_images.popleft()
        img = cv2.blur(img, (3, 3))
        old_images.append(img)

        img_gray, old_img_gray, match, old_template, absdiff = tm.find_template(old_images, (0, 400, 200))
        cv2.imshow('img_gray', img_gray)
        cv2.imshow('old_img_gray', old_img_gray)
        cv2.imshow('match', match)
        cv2.imshow('old_template', old_template)
        cv2.imshow('absdiff', absdiff)

#        cv2.imshow(
#            'img',
#            np.concatenate((img, img), axis=1))

#        dst = regress_perspecive(img, pts1, (height, width))
#        dst = cv2.addWeighted(frame, 0.3, dst, 0.7, 0)
#        cv2.imshow(
#           'inv', dst)

#        out.write(np.concatenate((img, img), axis=1))

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
