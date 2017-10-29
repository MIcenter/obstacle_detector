#!/usr/bin/python3

import numpy as np
import cv2


def make_np_array_from_points(key_points):
    return np.asarray([[i.pt] for i in key_points], dtype='float32')


def video_test(input_video_path=None, output_video_path=None):
    cx = 595
    cy = 303

    cap = cv2.VideoCapture(
        input_video_path \
            if input_video_path is not None \
            else input('enter video path: '))
    ret, frame = cap.read()
    for i in range(40 * 15):
        ret, frame = cap.read()

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path \
            if output_video_path is not None \
            else 'output.avi',
        fourcc, 15.0, (width * 4, height))

    ret, old_frame = cap.read()
    old_frame = old_frame[200:, 300:-300]
    old_frame = cv2.pyrDown(old_frame)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(old_frame)

    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    # create BFMatcher object

    while(ret):
        ret, frame = cap.read()
        ret, frame = cap.read()
        frame = frame[200:, 300:-300]
        frame = cv2.pyrDown(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow

        kp, _ = orb.detectAndCompute(old_gray,None)
        kp = make_np_array_from_points(kp)

        #img = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

        new_kp, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, kp, None)
        # Select good points
        good_new = new_kp[st==1]
        good_old = kp[st==1]
        # draw the tracks
        dists = []
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            dist = (c - a) ** 2 + (b - d) ** 2
            dists.append([a, b, dist])
#            mask = cv2.line(mask, (a,b),(c,d), (0, dist * 200, 0), 2)
        max_dist = max(dists, key=lambda dist: dist[2])[2]
        min_dist = min(dists, key=lambda dist: dist[2])[2]
        dists = map(
            lambda dist:
                [
                    dist[0],
                    dist[1],
                    (dist[2] - min_dist) ** 0.5 * 255 / max_dist ** 0.5],
            dists)
        dists = list(dists)
        knn_train_depth_data = []
        knn_responses_depth_data = []

        for i in dists:
            a, b, dist = i
            knn_train_depth_data.append((a, b))
            knn_responses_depth_data.append(dist)
            dist = int(dist)
            frame = cv2.circle(frame,(a,b),5 , (dist, 0, 0),-1)

        knn = cv2.ml.KNearest_create()
        knn_train_depth_data = np.asarray(
            knn_train_depth_data, dtype=np.float32)
        knn_responses_depth_data = np.asarray(
            knn_responses_depth_data, dtype=np.float32)
        knn.train(
            knn_train_depth_data,
            cv2.ml.ROW_SAMPLE,
            knn_responses_depth_data)
        height, width = mask.shape[:2]
        for y, line in enumerate(mask):
            for x, pixel in enumerate(line):
                newcomer = np.asarray([[x, y]], dtype=np.float32)
                ret, results, neighbours ,dist = knn.findNearest(newcomer, 1)
                value = results[0, 0] if dist[0, 0] < 10 else 0
                mask = cv2.circle(
                    mask, (x, y), 1 ,
                    (0, int(results[0, 0]), 0),-1)

        img = cv2.add(frame,mask)

        cv2.imshow('img',img)
        cv2.imshow('mask',mask)

        old_frame = frame.copy()
        old_gray = frame_gray.copy()
        mask = np.zeros_like(old_frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        elif k == ord('p'):
            cv2.waitKey()
        elif k == ord('s'):
            cv2.imwrite('screen.png', img)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

video_test('../../video/6.mp4', '../results/orb_out.avi')
