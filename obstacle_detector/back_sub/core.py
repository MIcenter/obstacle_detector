import cv2
import numpy as np

from functools import reduce


def get_shift_values(frames):
    lk_params = dict(
        winSize  = (5, 5), maxLevel = 2,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    gray_frames = list(
        map(
            lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            frames))

    p0 = np.float32([[100, 700]])
    values = [np.float32([[0, 0]])]
    for i in range(len(gray_frames) - 1):
        frame = gray_frames[i + 1]
        old_frame = gray_frames[i]
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_frame, frame,
            p0, None, **lk_params)
        values.append(p1 - p0)
    
    return values


def shift_image(img, x_shift, y_shift):
    height, width, _ = img.shape
    M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    dst = cv2.warpAffine(img, M, (width, height))
    return dst


def calc_diff(frames, shift_per_frame=0, frames_count=1):
    count = len(frames)
    if shift_per_frame == 0:
        shift_values = get_shift_values(frames)
        shift_values = map(
            lambda shift: shift[0][1],
            shift_values)
        shift_values = np.cumsum(list(shift_values))
        shift_values = list(reversed(shift_values))

        rolled_images = [
            shift_image(
                blured_image, 0, shift_values[i])
            for i, blured_image in enumerate(frames)]
    else:
        rolled_images = [
            shift_image(
                blured_image, 0, (count - i) * shift_per_frame // (count - 1))
            for i, blured_image in enumerate(frames)]

    old = rolled_images[-count]
    new = rolled_images[-1]
    new = cv2.cvtColor(new, cv2.COLOR_BGR2GRAY)
    result = np.zeros_like(new, dtype=np.int32)

    for old_img in rolled_images:
        old_img = cv2.cvtColor(old_img, cv2.COLOR_BGR2GRAY)
        result += cv2.absdiff(new, old_img)

    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    result = cv2.bitwise_and(
        result, result,
        mask=cv2.inRange(result, 32, 255))

    result = result.astype(new.dtype)
    new = cv2.cvtColor(new, cv2.COLOR_GRAY2BGR)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    return new, old, result
