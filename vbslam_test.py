#!/usr/bin/python3

import numpy as np
import cv2

from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.distance_calculator import spline_dist
from obstacle_detector.points_movement import make_np_array_from_points
from obstacle_detector.points_movement import make_points_from_np_array
from obstacle_detector.tm.image_shift_calculator import find_shift_value

from obstacle_detector.utils.keyboard_interface import handle_keyboard
from obstacle_detector.utils.gabor_filter import gabor_filter


def unite_points(img, iterations):
    for i in range(1 + iterations):
        for it in range(i):
            img = cv2.pyrDown(img)
        for it in range(i):
            img = cv2.pyrUp(img)
    return img





def create_mask_from_points_motion(
        mask, good_new, good_old,
        plane_shifting,
        M, center, height_in_pixels):

    'Now it uses few different and simple but effective methods'

    # initialize physical and computing parameters
    height, width = mask.shape[:2]
    cx, cy = center
    dx, dy, dz = plane_shifting

    transformed_new = cv2.perspectiveTransform(np.asarray([good_new]), M)[0]
    transformed_old = cv2.perspectiveTransform(np.asarray([good_old]), M)[0]

    dists = []
    x_shifted = []
    radiuses = set()
    for good_new_p, good_old_p, transformed_new_p, transformed_old_p in zip(
            good_new, good_old, transformed_new, transformed_old):

        x, y = good_new_p.ravel()
        x_old, y_old = good_old_p.ravel()

        a, b = transformed_new_p.ravel()
        c, d = transformed_old_p.ravel()

        if y < cy:
            continue
        # filter false positive movements
#        if b < d:
#            if y < cy:
#                continue
#                #dist = ((c - a) ** 2 + (b - d) ** 2 + height_in_pixels ** 2) ** 0.5
#                #dists.append([x, y, dist])
#                mask = cv2.circle(mask, (x, y), 4, (0, 255, 255), -1)
#            else:
#                #mask = cv2.circle(mask, (x, y), 5, (255, 0, 0), -1)
#                mask = cv2.circle(mask, (x, y), 4, (0, 255, 255), -1)
#            continue
        if abs(c-a) > 1:
            radius = max(1, ((x - x_old) ** 2 + (y - y_old) ** 2) ** 0.5)
            radiuses.add(int(radius))
            mask = cv2.circle(mask, (x, y), int(radius), (255, 0, 0), -1)
            x_shifted.append((x, y, radius, (a, b, c, d)))
            continue

        dist = ((c - a) ** 2 + (b - d) ** 2) ** 0.5
        dists.append([x, y, dist])

    # draw all points in green to see differents between obstacles and plane
    for pt in dists:
        x, y = pt[:2]
        mask = cv2.circle(mask, (x, y), 1, (0, 255, 0), -1)

    dists = filter(
        lambda dist:
            abs(
                (abs(dx) ** 2 + 
                abs(dy) ** 2) ** 0.5 - dist[2]) > 15 or
            dist[1] < cy,
        dists)
    dists = list(dists)

    if len(dists) == 0:
        return mask, x_shifted

    max_dist = max(dists, key=lambda dist: dist[2])[2]
    min_dist = min(dists, key=lambda dist: dist[2])[2]

    dists = map(
        lambda dist:
            [
                dist[0],
                dist[1],
                (dist[2] - min_dist) ** 0.5 * 255 / max_dist ** 0.5],
        dists)

    for i in dists:
        a, b, dist = i
        dist = int(dist)
        mask = cv2.circle(mask,(a,b),2 , (0, 0, 255),-1)

    return mask, x_shifted


def video_test(input_video_path=None, output_video_path=None):
    # physical and computing parameters
    cx = 595 - 300
    cy = 303 - 200

    roi_width = 6
    roi_length = 20
    roi_height = 4.4

    # video output block
    cap = cv2.VideoCapture(
        input_video_path \
            if input_video_path is not None \
            else input('enter video path: '))
    ret, frame = cap.read()

    out_height, out_width, _ = frame[200:, 300:-300].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path \
            if output_video_path is not None \
            else 'output.avi',
        fourcc, 15.0, (out_width * 2, out_height))

    # initialize old_frames
    old_frame = cap.read()[1][200:, 300:-300]

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(old_frame)

    old_img = cv2.add(old_frame, mask)
    old_transformed_frame, pts1, M = inv_persp_new(
            old_img, (cx, cy),
            (roi_width, roi_length), spline_dist, 200)

    # orb initialize
    orb = cv2.ORB_create(nfeatures=3500)

    # main loop #TODO fps output
    frame = old_frame.copy()
    while(True):
        for i in range(4):
            frame = cap.read()[1][200:, 300:-300]

        #frame = cv2.pyrUp(cv2.pyrDown(frame))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transformed_frame, pts1, M = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), spline_dist, 200)

        kp = make_np_array_from_points(orb.detectAndCompute(old_gray,None)[0])

        new_kp, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, kp, None)

        dx, dy = find_shift_value(
            transformed_frame, old_transformed_frame, (50, 300, 250, 550))

        mask, x_shifted = create_mask_from_points_motion(
            mask, new_kp[st==1], kp[st==1],
            (dx, dy, 4.4),
            M, (cx, cy), roi_height * old_frame.shape[1] / roi_width)

        img = cv2.add(frame,mask)

        transformed_img, pts1, M = inv_persp_new(
            img, (cx, cy), (roi_width, roi_length), spline_dist, 200)

        handled_mask = cv2.pyrDown(mask.copy())

        opening = cv2.morphologyEx(
            handled_mask[..., 0], cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8), iterations=3)
        opening = cv2.inRange(opening, 1, 255)
        drawed_contours = np.zeros_like(handled_mask)
        ret, thresh = cv2.threshold(opening, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        drawed_contours = cv2.drawContours(
            drawed_contours, contours, -1, (255,0, 255), 1)
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            drawed_contours = cv2.rectangle(
                drawed_contours, (x,y),(x+w,y+h),(0,0,255),1)


        handled_mask[...,1] = 0
        handled_mask[...,2] = 0#cv2.equalizeHist(handled_mask[...,2])

        cv2.imshow('mask', np.concatenate(
            (
                handled_mask,
                cv2.cvtColor(
                    opening,
                    cv2.COLOR_GRAY2BGR),
                drawed_contours
            ), axis=1))

#        cv2.imshow('out', np.concatenate((img, mask), axis=1))
        cv2.imshow('img', cv2.addWeighted(
            frame, 0.5,
            cv2.pyrUp(drawed_contours), 0.5,
            0))
#        out.write(np.concatenate((img, mask), axis=1))

        old_frame = frame.copy()
        old_gray = frame_gray.copy()
        old_img = img.copy()
        old_transformed_frame = transformed_frame.copy()
        mask = np.zeros_like(old_frame)

        if handle_keyboard(screenshot_image=None) == 1:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_test('../../video/1.mp4', '../results/motion_of_pointorb_out.avi')
