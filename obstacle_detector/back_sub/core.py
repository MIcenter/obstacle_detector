import cv2
import numpy as np

def calc_diff(frames, shift_per_frame=0, frames_count=1):
    count = len(frames)
    rolled_images = [
        np.roll(blured_image, (count - i) * shift_per_frame // (count - 1), axis=0)
            for i, blured_image in enumerate(frames)]

    old = rolled_images[-count]
    new = rolled_images[-1]
    result = np.zeros_like(new, dtype=np.int32)

    for old_img in rolled_images:
        result += cv2.absdiff(new, old_img)

    result //= count
    result *= 3
    result = result.astype(new.dtype)

    return new, old, result
