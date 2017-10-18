import cv2
import numpy as np

from ..points_movement import *


def find_sift_matches(gray1, gray2, shift_distance):
    sift = cv2.SIFT()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1,None)
    kp2, des2 = sift.detectAndCompute(gray2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.
    gray3 = cv2.drawMatchesKnn(gray1,kp1,gray2,kp2,good,flags=2)
    return gray3


def find_orb_matches(gray1, gray2, shift_distance):
    orb = cv2.ORB_create(edgeThreshold=5)

#    gray1 = cv2.bilateralFilter(gray1, 9, 75, 75)
 #   gray2 = cv2.bilateralFilter(gray2, 9, 75, 75)

    gray1 = cv2.equalizeHist(gray1)
    gray2 = cv2.equalizeHist(gray2)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return np.concatenate((gray1, gray2), axis=1)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING,  crossCheck=True)

    matches = bf.match(des1, des2)

    matches = list(matches)
    distances = map(lambda m: m.distance, matches)

    matches = filter(lambda x: x.distance < 20, matches)
    matches = list(matches)
    print(len(matches))


    def create_movement_vector(match, kp1=kp1, kp2=kp2):
        pt1 = kp2[match.trainIdx]
        pt2 = kp1[match.queryIdx]
        return \
            abs(pt1.pt[0] - pt2.pt[0]),\
            abs(pt1.pt[1] - pt2.pt[1])

    movements = map(create_movement_vector, matches)
    def filter_obstacle_movement(match):
        x, y = create_movement_vector(match)
        return 1 <= y <= 5 and x < 5

    matches = filter(filter_obstacle_movement, matches)
    matches = list(matches)

    obstacle_kps = [kp1[i.queryIdx] for i in matches]
    gray1 = cv2.drawKeypoints(gray1, obstacle_kps, None, color=(0,0,255), flags=0)
    # gray1 = cv2.drawKeypoints(gray1, kp1, None, color=(0,255,0), flags=0)
    # gray2 = cv2.drawKeypoints(gray2, kp2, None, color=(0,255,0), flags=0)

    gray3 = cv2.drawMatches(gray1, kp1, gray2, kp2, matches[0:0], None, flags=2)
    return gray3


def calc_dense_optical_flow(img, old_img):
    height, width = img.shape
    hsv = np.zeros([height, width, 3], dtype=np.uint8)
    hsv[...,1] = 255
    flow = cv2.calcOpticalFlowFarneback(old_img, img, None, 0.5, 1, 2, 10, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    dist = hsv[...,2]
    mask = cv2.inRange(dist, 16, 255)
    hsv[...,2] = cv2.bitwise_and(dist, dist, mask=mask)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr, dist


def calc_sparse_optical_flow(img, old_img, points=None):
    if points is None:
        points = find_features(img)

    shift_distance = 15
    img = cv2.equalizeHist(img)
    old_img = cv2.equalizeHist(old_img)
    points_moves = get_points_moves(points, img, old_img, shift_distance)

    points_moves = filter(lambda mv: abs(mv[0][0] - mv[1][0]) < 2, points_moves)
    points_moves = filter(lambda mv: 15 < mv[2] < 75, points_moves)
    points_moves =  list(points_moves)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    old_img = cv2.cvtColor(old_img, cv2.COLOR_GRAY2BGR)
    mask = np.zeros_like(img)
    for i in points_moves:
        img = cv2.line(img, i[0], i[1], (255, 0, 0), int(i[2] // 15))
        mask = cv2.line(mask, i[0], i[1], (255, 0, 0), int(i[2] // 15))

    return img, mask


def get_square_from_img(img, rectangle):
    x, y, width, height = rectangle
    x1, y1, x2, y2 = x, y, x + width, y + height
    return img[y1:y2, x1:x2].copy()


def find_obstacles_between_two_images(img, old_img, coords, method=None):
    old_template = get_square_from_img(old_img, coords)

    res = cv2.matchTemplate(img, old_template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc

    t_height, t_width = old_template.shape

    bottom_right = (top_left[0] + t_width, top_left[1] + t_height)
    original_bottom_right = (coords[0] + coords[2], coords[1] + coords[2])
    (x1, y1), (x2, y2) = top_left, bottom_right

    match = img[y1:y2, x1:x2].copy()

    absdiff = cv2.absdiff(old_template, match)
    absdiff = cv2.equalizeHist(absdiff)

    shift_distance = y1 - coords[1] -2
    matches = None
    if method is None or method == 'orb':
        matches, mask = find_orb_matches(match, old_template, shift_distance)
    elif method == 'sift':
        matches, mask = find_sift_matches(match, old_template, shift_distance)
    elif method == 'sparse':
        matches, mask = calc_sparse_optical_flow(match, old_template)
        cv2.imshow('mask', mask)
    elif method == 'dense':
        matches, mask = calc_dense_optical_flow(match, old_template)

    return matches, absdiff, mask


def find_obstacles(frames, coords, method=None):
    img = frames[-1].gray.copy()

    result_mask = np.zeros([400, 200, 3], dtype=np.float32)
    result_matches = np.zeros([400, 200, 3], dtype=np.float32)
    for i in range(0, len(frames) - 1, 2):
        old_img = frames[i].gray.copy() 
        matches, absdiff, mask = find_obstacles_between_two_images(
            img, old_img, coords, method)
        result_mask += mask
        result_matches += matches

    result_mask /= (len(frames) - 1) // 4
    result_matches /= (len(frames) - 1) // 2
    result_mask.astype(np.uint8)
    result_matches.astype(np.uint8)

    return img, old_img, matches, absdiff
