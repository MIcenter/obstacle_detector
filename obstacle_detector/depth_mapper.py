import numpy as np
import cv2

def calculate_depth_map(L, R):
    imgL = cv2.pyrDown(L)  # downscale images for faster processing
    imgR = cv2.pyrDown(R)

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    res = (disp - min_disp) / num_disp
    cv2.imshow('real_depth', res)

    max_value = max(res.ravel())
    min_value = min(res.ravel())
    # res += min_value
    # res /= max_value
    # res *= 255
    res = cv2.normalize(res, 0, 255, cv2.NORM_L1)
    res = res.astype(np.uint8)

    return res
