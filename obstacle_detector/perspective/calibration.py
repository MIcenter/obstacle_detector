import cv2
import numpy as np
import csv

from .perspective_transformer import inv_persp_new
from ..distance_calculator import Distance_calculator


def get_lines_intersection(line_1, line_2):
    x1, y1, x2, y2 = line_1
    A1, B1, C1 = y1 - y2, x2 - x1, x1 * y2 - x2 * y1

    x1, y1, x2, y2 = line_2
    A2, B2, C2 = y1 - y2, x2 - x1, x1 * y2 - x2 * y1

    center_x = -(C1 * B2 - C2 * B1) / (A1 * B2 - A2 * B1)
    center_y = -(A1 * C2 - A2 * C1) / (A1 * B2 - A2 * B1)

    return center_x, center_y


def get_lanes_intersection(img, roi):
    left_up_x, left_up_y, right_up_x, right_up_y = roi
    img_roi = img[
        left_up_y : right_up_y,
        left_up_x : right_up_x]

    # magic
    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    lines = filter(
        lambda line:
            abs(line[0][1] - line[0][3]) / abs(line[0][0] - line[0][2]) > 4,
        lines)

    lines = list(lines)

    left = max(
        lines,
        key=lambda line:
            0 if line[0][1] < line[0][3] else abs(line[0][1] - line[0][3]))[0]

    right = max(
        lines,
        key=lambda line:
            0 if line[0][1] > line[0][3] else abs(line[0][1] - line[0][3]))[0]

    center_x, center_y = get_lines_intersection(left, right)

    x1, y1, x2, y2 = left
    cv2.line(img_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

    x1, y1, x2, y2 = right
    cv2.line(img_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

    img_roi[int(center_y), int(center_x)] = (255, 0, 0)

    return img_roi, center_x + left_up_x, center_y + left_up_y


def find_center_point(frames, roi):
    avg_cx = avg_cy = i = 0
    for i, frame in enumerate(frames):
        _, cx, cy = get_lanes_intersection(frame, roi)
        avg_cx += cx
        avg_cy += cy
    i += 1

    return avg_cx // i, avg_cy // i


def calibrate_center(new_frame, prev_frame, center, roi, expected_diff=None):
	with open('data/spline-data.csv') as csv_file:
		spline_data = csv.reader(csv_file)
		pxs, meters = zip(*spline_data)
		pxs = list(map(lambda x: int(x), pxs))
		meters = list(map(lambda x: float(x), meters))

	distance_calculator = Distance_calculator(pxs, meters)

	cx, cy = center
	roi_width, roi_length = roi

	new_img, pts1 = inv_persp_new(
		new_frame, (cx, cy),
        (roi_width, roi_length), distance_calculator, 200)

	prev_img, pts1 = inv_persp_new(
		prev_frame, (cx, cy),
        (roi_width, roi_length), distance_calculator, 200)

	lk_params = dict(
		winSize  = (15,15),
		maxLevel = 2,
		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	old_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
	img_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

	points_old = np.asarray([[
		[100, 3 * img_gray.shape[0] // 10],
		[100, 9 * img_gray.shape[0] // 10]]], dtype=np.float32)
	points, st, err = cv2.calcOpticalFlowPyrLK(
		old_gray, img_gray, points_old, None, **lk_params)

	y_dist_diff = \
		(points[0][0][1] - points_old[0][0][1]) - \
		(points[0][1][1] - points_old[0][1][1])

	if expected_diff is None:
		if y_dist_diff < 0:
			return \
				calibrate_center(
					new_frame, prev_frame,
					(cx, cy + 1), roi, expected_diff=y_dist_diff)
		else:
			return \
				calibrate_center(
					new_frame, prev_frame,
					(cx, cy - 1), roi, expected_diff=y_dist_diff)
	elif expected_diff < 0:
		if y_dist_diff < 0:
			return \
				calibrate_center(
					new_frame, prev_frame,
					(cx, cy + 1), roi, expected_diff=y_dist_diff)
		else:
			return \
				(cx, cy) if abs(y_dist_diff) < abs(expected_diff) else \
				(cx, cy - 1)
	else:
		if y_dist_diff > 0:
			return \
				calibrate_center(
					new_frame, prev_frame,
					(cx, cy - 1), roi, expected_diff=y_dist_diff)
		else:
			return \
				(cx, cy) if abs(y_dist_diff) < abs(expected_diff) else \
				(cx, cy + 1)

	return center
