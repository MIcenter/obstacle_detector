import cv2
import numpy as np
import csv

from .perspective_transformer import inv_persp_new
from ..distance_calculator import Distance_calculator

def find_center_point(new_img, prev_img, old_pts, eps=0.1):
	lk_params = dict(
		winSize  = (35,35),
		maxLevel = 2,
		criteria = (
			cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.003))

	old_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
	new_gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)


	pts, st, err = cv2.calcOpticalFlowPyrLK(
		new_gray, old_gray, old_pts, None, **lk_params)

	print(old_pts)
	print(pts)
	# Select good points
	good_new = pts[st==1]
	good_old = old_pts[st==1]

	a, b = good_new.ravel()
	c, d = good_old.ravel()

	s = ((a - c) ** 2 + (b - d) ** 2) ** 0.5
	print(s)
	if (s > eps):
		return find_center_point(new_img, prev_img, pts, eps)
	else:
		return pts


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
		new_frame, (cx, cy), (roi_width, roi_length), distance_calculator, 200)

	prev_img, pts1 = inv_persp_new(
		prev_frame, (cx, cy), (roi_width, roi_length), distance_calculator, 200)

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

	print(y_dist_diff)
	print(points)

	print('center:', cx, cy)
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
