#!/usr/bin/python3

import numpy as np
import cv2

from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.points_movement import make_np_array_from_points

from obstacle_detector.utils.keyboard_interface import handle_keyboard
from obstacle_detector.utils.sum_maps_equal import sum_maps_equal

from obstacle_detector.homography import calculate_obstacles_map

from tools.decode_video_from_json import decode_stdin


def video_test(output_video_path=None):
    cx = 595 - 300
    cy = 303 - 200

    roi_width = 6
    roi_length = 20
    roi_height = 4.4
    px_height_of_roi_length = 352

    # video output block
    json_frames = decode_stdin()
    frame = json_frames.__next__()

    out_height, out_width, _ = frame[200:, 300:-300].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path \
            if output_video_path is not None \
            else 'output.avi',
        fourcc, 15.0, (out_width, out_height))

    # initialize old_frames
    old_frame = frame[200:, 300:-300]

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(old_frame)

    old_img = cv2.add(old_frame, mask)
    old_transformed_frame, pts1, M = inv_persp_new(
            old_img, (cx, cy),
            (roi_width, roi_length), px_height_of_roi_length, 400)

    # orb initialize
    orb = cv2.ORB_create(nfeatures=3500)

    # main loop #TODO fps output
    for frame_number, frame in enumerate(json_frames):
        frame = frame[200:, 300:-300]

        #frame = cv2.pyrUp(cv2.pyrDown(frame))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transformed_frame, pts1, M = inv_persp_new(
            frame, (cx, cy), (roi_width, roi_length), px_height_of_roi_length, 400)

        kp = make_np_array_from_points(orb.detectAndCompute(old_gray,None)[0])

        new_kp, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, kp, None)

        masks, drawed_contours, obstacles_blocks_list = calculate_obstacles_map(
            mask, new_kp[st==1], kp[st==1], M, (cx, cy))

        img = frame.copy()
        obstacles = sum_maps_equal(
            obstacles_blocks_list,
            [1 for i in range(len(obstacles_blocks_list))])
        obstacles = cv2.bitwise_and(
            frame, frame,
            mask=cv2.inRange(cv2.pyrUp(obstacles), (0, 0, 1), (0, 0, 255)))

        cv2.imshow(
            'obstacles',
            cv2.addWeighted(
                frame, 0.3, obstacles, 0.7, 0))
#        cv2.imshow('img', cv2.addWeighted(
#            img, 0.5,
#            cv2.pyrUp(drawed_contours[3]), 0.5,
#            0))
        out.write(
            cv2.add(
                frame, obstacles))

        old_frame = frame.copy()
        old_gray = frame_gray.copy()
        old_img = img.copy()
        old_transformed_frame = transformed_frame.copy()
        mask = np.zeros_like(old_frame)

        if handle_keyboard(screenshot_image=None) == 1:
            break

    out.release()
    cv2.destroyAllWindows()

video_test('../results/motion_of_pointorb_out_2_united.avi')
