#!/usr/bin/python3

import numpy as np
import cv2

from obstacle_detector.perspective import inv_persp_new
from obstacle_detector.points_movement import make_np_array_from_points

from obstacle_detector.utils.keyboard_interface import handle_keyboard
from obstacle_detector.utils.sum_maps_equal import sum_maps_equal

from obstacle_detector.homography import calculate_obstacles_map

from tools.decode_video_from_json import decode_stdin


def find_template_in_frame(template, frame):
    res = cv2.matchTemplate(frame, template,cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    x, y = min_loc

    return x, y



def update_tracking_rectangle(trackers, frame, old_frame):
    res = []
    height, width = frame.shape[:2]
    height //= 2
    width //= 2

    for t in trackers:
        x_center, y_center, rect, time, area, max_val, mean_val = t
        x, y, w, h = rect

        x *= 2
        y *= 2
        w *= 2
        h *= 2

        template = old_frame[y:y+h,x:x+h]
        x, y = find_template_in_frame(template, frame)

        x //= 2
        y //= 2
        h //= 2
        w //= 2

        rect = x, y, w, h
        t = x_center, y_center, rect, time, area, max_val, mean_val
        if 0 <= x_center < width and 0 <= y_center < width:
            res.append(t)
    return res


def track_obstacles(detected_obstacles, last_trackers):
    result = detected_obstacles.copy()
    imgray = cv2.cvtColor(detected_obstacles, cv2.COLOR_BGR2GRAY)
    thresh = cv2.inRange(detected_obstacles, (0, 0, 1), (255, 255, 255))

    im2, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    new_trackers = []
    def is_tracker_new(tracker):
        for t in last_trackers:
            x_center, y_center = tracker[:2]
            if abs(x_center - t[0]) < t[2][2] and\
               abs(y_center - t[1]) < t[2][3]:
                return False
        return True


    def find_tracker_in_last(tracker):
        for t in last_trackers:
            x_center, y_center = tracker[:2]
            if abs(x_center - t[0]) < t[2][2] and\
               abs(y_center - t[1]) < t[2][3]:
                return t


    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        mask = np.zeros(imgray.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
        pixelpoints = np.transpose(np.nonzero(mask))
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray,mask = mask)
        mean_val = cv2.mean(detected_obstacles, mask = mask)
        area = cv2.contourArea(cnt)
        rectangle_area = w * h

        if area < 25:
            continue
        time = 1

        t = (
            x + w, y + w,
            (x, y, w, h),
            time,
            area, max_val, mean_val)

        if len(pixelpoints) / rectangle_area > 0.8 and is_tracker_new(t):
            cv2.rectangle(result,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(
                result,
                (x + w // 2, y + h // 2), 1,
                (255,255,255))
            new_trackers.append(t)

    for t in last_trackers:
        x,y,w,h = t[2]
        mean_val = t[6]
        cv2.rectangle(result,(x,y),(x+w,y+h),(0,0,(mean_val[2]+10)),2)
        cv2.circle(
            result,
            (x + w // 2, y + h // 2), 1,
            (255,255,255))


    new_trackers = [i for i in filter(is_tracker_new, new_trackers)]
    print('new:',len(new_trackers))
    return result, new_trackers + last_trackers


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
    orb = cv2.ORB_create(nfeatures=5500, edgeThreshold=10)

    # main loop #TODO fps output
    trackers = []

    for frame_number, frame in enumerate(json_frames):
        key = handle_keyboard(screenshot_image=None)
        if key == 1:
            break
        elif key == ord('n'):
            trackers = []
            continue

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
        detected_obstacles = sum_maps_equal(
            obstacles_blocks_list,
            [1 for i in range(len(obstacles_blocks_list))])
        obstacles = cv2.bitwise_and(
            frame, frame,
            mask=cv2.inRange(
                cv2.pyrUp(detected_obstacles), (0, 0, 1), (0, 0, 255)))

        cv2.imshow(
            'tracked_obstacles',
            cv2.addWeighted(
                frame, 0.3, obstacles, 0.7, 0))

        trackers = update_tracking_rectangle(trackers, frame, old_frame)
        detected_obstacles, trackers = track_obstacles(detected_obstacles, trackers)

        print('all trackers:', len(trackers))
        print([i[4] for i in trackers])

        cv2.imshow(
            'detected obstacles',
            cv2.addWeighted(
                frame, 0.4, cv2.pyrUp(detected_obstacles), 0.6, 0))

        out.write(
            cv2.addWeighted(
                frame, 0.4, cv2.pyrUp(detected_obstacles), 0.6, 0))

        old_frame = frame.copy()
        old_gray = frame_gray.copy()
        old_img = img.copy()
        old_transformed_frame = transformed_frame.copy()
        mask = np.zeros_like(old_frame)

    out.release()
    cv2.destroyAllWindows()

video_test('../results/motion_of_pointorb_out_2_united.avi')
