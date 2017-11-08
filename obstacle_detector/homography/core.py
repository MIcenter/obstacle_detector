import cv2
import numpy as np


def create_point_shift_structure(
        good_new, good_old, M):

    """TODO doc"""

    transformed_new = cv2.perspectiveTransform(np.asarray([good_new]), M)[0]
    transformed_old = cv2.perspectiveTransform(np.asarray([good_old]), M)[0]

    dists = []

    for good_new_p, good_old_p, transformed_new_p, transformed_old_p in zip(
            good_new, good_old, transformed_new, transformed_old):

        x, y = good_new_p.ravel()
        x_old, y_old = good_old_p.ravel()

        a, b = transformed_new_p.ravel()
        c, d = transformed_old_p.ravel()

        dists.append(((x, y), (x_old, y_old), (a, b, c, d)))

    return dists


def find_homography_anomalies(shifted_points):

    """Find points in shifted points where x coord has been changed"""

    x_shifted_points = []

    for pt in shifted_points:
        (x, y), (x_old, y_old), (a, b, c, d) = pt

        if abs(c-a) > 2:
            radius = max(1, ((x - x_old) ** 2 + (y - y_old) ** 2) ** 0.5)
            x_shifted_points.append((x, y, radius))

    return x_shifted_points


def create_mask_from_points_motion(
        mask, good_new, good_old,
        M, center):

    'Now it uses few different and simple but effective methods'

    # initialize physical and computing parameters
    cx, cy = center

    shifted_points = create_point_shift_structure(good_new, good_old, M)
    x_shifted_points = find_homography_anomalies(shifted_points)
    x_shifted_points = list(filter(
        lambda pt:
            pt[1] > cy and pt[2] < 10,
        x_shifted_points))

    little_shifts = filter(
        lambda pt:
            1 <= pt[2] <= 1,
        x_shifted_points)

    middle_shifts = filter(
        lambda pt:
            2 <= pt[2] <= 4,
        x_shifted_points)

    big_shifts = filter(
        lambda pt:
            3 <= pt[2] <= 5,
        x_shifted_points)

    large_shifts = filter(
        lambda pt:
            4 <= pt[2] <= 9,
        x_shifted_points)

    shifts = [
        little_shifts,
        middle_shifts,
        big_shifts,
        large_shifts]

    masks = []

    for points in shifts:
        new_mask = np.zeros_like(mask)
        for pt in points:
            x, y, radius = pt
            new_mask = cv2.circle(new_mask,(x,y), int(radius), (255, 0, 0),-1)

        masks.append(new_mask)

    return masks, x_shifted_points


def calculate_obstacles_map(
        mask, new_kp, kp, M, center):

    """TODO DOCS"""

    masks, x_shifted = create_mask_from_points_motion(
        mask, new_kp, kp,
        M, center)

    drawed_contours_list = []
    obstacles_blocks_list = []

    for i, mask in enumerate(masks):
        mask = cv2.pyrDown(mask.copy())

        opening = cv2.morphologyEx(
            mask[..., 0], cv2.MORPH_CLOSE,
            np.ones((5, 5), dtype=np.uint8), iterations=1+i)

        opening = cv2.inRange(opening, 1, 255)

        drawed_contours = np.zeros_like(mask)
        obstacles_blocks = np.zeros_like(mask)

        ret, thresh = cv2.threshold(opening, 127, 255, 0)
        im2, contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        drawed_contours = cv2.drawContours(
            drawed_contours, contours, -1, (255,0, 255), 1)

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            obstacles_blocks = cv2.rectangle(
                obstacles_blocks, (x, y), (x + w, y + h),
                (0, 0, 255), thickness=cv2.FILLED)
            drawed_contours = cv2.rectangle(
                drawed_contours, (x,y),(x+w,y+h),(0,0,255),1)

        drawed_contours_list.append(drawed_contours)
        obstacles_blocks_list.append(obstacles_blocks)

    return masks, drawed_contours_list, obstacles_blocks_list
