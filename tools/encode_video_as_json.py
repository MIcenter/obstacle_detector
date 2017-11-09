#!/usr/bin/python3

import cv2
import numpy as np

import json
import sys


def video_as_json(path=None):
    if path is None:
        path = 0

    cap = cv2.VideoCapture(path)

    ret = True
    timestamp = 0

    while(ret):
        ret, frame = cap.read()
        if ret == False:
            break

        height = frame.shape[0]
        width = frame.shape[1]
        size = frame.size
        stride = 0
        timestamp += 1
        format_prop = 'bgr24'

        frame_info = {
            "width" : width,
            "height" : height,
            "size" : size,
            "stride" : stride,
            "timestamp" : timestamp,
            "format" : format_prop,
            "dtype" : str(frame.dtype)
        }

        yield json.dumps(frame_info)
        yield frame.reshape(-1).tobytes().hex()


def print_video_as_json(path=None):
    for i, line in enumerate(video_as_json(path)):
        print(line)

print_video_as_json(input())
