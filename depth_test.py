#!/usr/bin/python3

import numpy as np
import cv2

from collections import deque

from obstacle_detector.distance_calculator import spline_dist
from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.perspective import regress_perspecive

from obstacle_detector.depth_mapper import calculate_depth_map
from obstacle_detector.tm.image_shift_calculator import find_shift_value

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
        original_frames.popleft()
        ret, frame = cap.read()
        original_frames.append(frame)

        img, pts1 = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)
        old_images.popleft()
        old_images.append(img)

        left = original_frames[-5][:, width // 2:]
        right = original_frames[-1][:, width // 2:]

        left = cv2.pyrDown(left)
        left = cv2.blur(left, (3, 3))
        right = cv2.pyrDown(right)
        right = cv2.blur(right, (3, 3))

        depth = calculate_depth_map(left, right)
        cv2.imshow('left', left)
        cv2.imshow('right', right)
        cv2.imshow('depth', depth)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        res = cv2.addWeighted(left, 0.5, depth, 0.5, 0)
        cv2.imshow('res', res)

#        left = old_images[-1][300:,:]
#        right = old_images[-9][300:,:]
#
#        shift_value = find_shift_value(left, right, (30, 100, 60, 300))
#        right = np.roll(right, shift_value[1], axis=0)#shift_value[0])
#        right = np.roll(right, shift_value[0], axis=1)#shift_value[0])
#        left = left[100:-100,:]
#        right = right[100:-100,:]
#
#        print(shift_value)
#
#        left = np.rot90(left, 3)
#        right = np.rot90(right, 3)
#
#        cv2.imshow('left', left)
#        cv2.imshow('right', right)
#
#        shifted_map = cv2.equalizeHist(
#            calculate_depth_map(
#                left, right))
#        cv2.imshow(
#            'shifted map', shifted_map)
#        diff = cv2.absdiff(left, right)
#        cv2.imshow('diff', diff)

#        dm = calculate_depth_map(left, right)
#        cv2.imshow('dm', dm)
#        dm = cv2.equalizeHist(dm)
#        cv2.imshow('eq dm', dm)

#        dm = cv2.cvtColor(dm, cv2.COLOR_GRAY2BGR)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('screen.png', img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_test('../../video/6.mp4', '../results/depth_map_out.avi')
