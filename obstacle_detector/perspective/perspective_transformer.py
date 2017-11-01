import cv2
import numpy as np

from functools import lru_cache

# inverse perspective mapping defines
def inv_persp_new(
        image, center, physical_roi,
        distance_calculator, out_width=100, out_height=None):
    'TODO DOC'
    cx, cy = center
    height, width, _ = image.shape
    roi_width, roi_length = physical_roi

    L = R = (0, 0)

    if cx < width // 2:
        L = (0, height - 1)
        R = (cx * 2, height - 1)
    else:
        L = (cx * 2 - width, height - 1)
        R = (width - 1, height - 1)

    far_height = int(
        distance_calculator.get_rails_px_height_by_distance(roi_length))

    far_L = ((cx - L[0]) * far_height // (height - cy), height - far_height)
    far_R = (cx * 2 - far_L[0], height - far_height)

    if out_height is None:
        out_height = out_width * roi_length // roi_width

    pts1 = np.float32([
        [L[0], L[1]],
        [R[0], R[1]],
        [far_L[0], far_L[1]],
        [far_R[0], far_R[1]]]) 
    pts2 = np.float32([
        [0, out_height],
        [out_width, out_height],
        [0, 0],
        [out_width, 0]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(image, M, (out_width, out_height))

    return dst, pts1, M


def regress_perspecive(img, pts1, shape):
    height, width = shape

    img_height, img_width, _ = img.shape
    pts2 = np.float32([
        [0, img_height],
        [img_width, img_height],
        [0, 0],
        [img_width, 0]])

    M = cv2.getPerspectiveTransform(pts2, pts1)

    dst = cv2.warpPerspective(img, M, (width, height))

    return dst
