#!/usr/bin/python3

import cv2
import numpy as np
import skimage
import sklearn

from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.distance_calculator import spline_dist

def video_test(input_video_path=None):
    cx = 603
    cy = 297
    roi_width = 25
    roi_length = 90

    px_height_of_roi_length = 352
    #int(
    #    spline_dist.get_rails_px_height_by_distance(roi_length))
    #print(px_height_of_roi_length)

    cap = cv2.VideoCapture(
        input_video_path \
            if input_video_path is not None \
            else input('enter video path: '))

    ret, frame = cap.read()

    while(ret):
        ret, frame = cap.read()

        transformed_plane, pts1, M = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length),
            px_height_of_roi_length, 200)

        extra_transformed_plane, pts1, M = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length),
            px_height_of_roi_length, 200,
            extra_width=200 * 2)

        cv2.imshow(
            'plane of the way',
            transformed_plane)

        cv2.imshow(
            'plane',
            extra_transformed_plane)

        cv2.imshow(
            'original frame',
            frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('screen.png', img)

    cap.release()
    cv2.destroyAllWindows()

video_test('../../video/6.mp4')
