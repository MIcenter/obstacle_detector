import cv2
import numpy as np

def calc_diff(frames, shift_per_frame=0, frames_count=1):
    count = len(frames)
    rolled_images = [
        np.roll(
            blured_image, (count - i) * shift_per_frame // (count - 1), axis=0)
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
