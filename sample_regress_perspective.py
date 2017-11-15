#!/usr/bin/python3

import cv2
import numpy as np

# Local Binary Pattern function
#from skimage.feature import local_binary_pattern

from obstacle_detector.perspective import inv_persp_new, regress_perspecive
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
            px_height_of_roi_length, 500)

        extra_transformed_plane, pts1, M = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length),
            px_height_of_roi_length, 200,
            extra_width=200 * 2)

        gray = cv2.cvtColor(extra_transformed_plane, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(
                extra_transformed_plane,
                (x1, y1), (x2, y2), (0, 255, 0), 2)

        regressed_image = regress_perspecive(
            extra_transformed_plane, pts1, frame.shape[:2], 400)

        cv2.imshow(
            'frame',
            cv2.addWeighted(
                regressed_image, 0.5,
                frame, 0.5,
                0))

        cv2.imshow(
            'plane',
            extra_transformed_plane)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('screen.png', extra_transformed_plane)

    cap.release()
    cv2.destroyAllWindows()

video_test('../../video/6.mp4')
