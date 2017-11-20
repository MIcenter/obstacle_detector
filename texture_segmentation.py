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
            px_height_of_roi_length, 200)

        extra_transformed_plane, pts1, M = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length),
            px_height_of_roi_length, 200,
            extra_width=200 * 2)

        gray = cv2.cvtColor(transformed_plane, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        for y in range(gray.shape[0]):
            for x in range(gray.shape[1]):
                gray[y, x] = cv2.fastAtan2(sobely[y, x], sobelx[y, x]) * 255 // 360
        s = v = np.ones_like(gray, dtype=np.uint8) * 255
        hsv = cv2.merge([gray, s, v])
        hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        height = hsv.shape[0]
        w, h = 10, 10
        #hsv = cv2.blur(hsv, (2, 2))
        template = hsv[height - h - 1: height - 1, 90:90 + w]
        res = cv2.matchTemplate(hsv, template,cv2.TM_SQDIFF_NORMED)
        threshold = 0.5
        loc = np.where( res <= 1-threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(hsv, pt, (pt[0] + w, pt[1] + h), (0,0,0), 2, cv2.FILLED)
        cv2.imshow('res', res)
#        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        hsv = cv2.rectangle(
            hsv,
            (90, hsv.shape[0] - 21), (110, hsv.shape[0] - 1),
            (0, 0, 0), 2)

        regressed_image = regress_perspecive(
            extra_transformed_plane, pts1, frame.shape[:2], 400)
        regressed_texture = regress_perspecive(
            hsv, pts1, frame.shape[:2], 0)

        cv2.imshow(
            'frame',
            cv2.addWeighted(
                regressed_texture, 0.5,
                frame, 0.5,
                0))

        cv2.imshow(
            'plane',
            hsv)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('screen2.png', regressed_image)

    cap.release()
    cv2.destroyAllWindows()

video_test('../../video/6.mp4')
