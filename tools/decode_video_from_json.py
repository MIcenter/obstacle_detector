#!/usr/bin/python3

import cv2
import numpy as np

import json
import sys


def decode_stdin():
    while not sys.stdin.closed:
        json_line = sys.stdin.buffer.read(1000)

        json_end = -1
        while len(json_line) and json_end == -1:
            json_end = json_line.find(0)
            if json_end == -1:
                json_line += sys.stdin.buffer.read(1000)

        if len(json_line) == 0:
            break

        json_str = json_line[:json_end].decode('utf-8')

        json_info = json.loads(json_str)

        size = int(json_info["img-size"])
        height = int(json_info["height"])
        width = int(json_info["width"])
        format_prop = json_info["format"]

        img_buf = json_line[json_end + 1:]
        img_buf += sys.stdin.buffer.read(size - len(img_buf))

        dtype = None
        channels = 1
        arr = None

        if format_prop == 'gray8':
            dtype = np.uint8
        elif format_prop == 'bgr24':
            dtype=np.uint8
            channels = 3

        if channels == 1:
            arr = np.frombuffer(
                img_buf[:size],
                dtype=dtype).reshape((height, width))
        elif channels == 3:
            arr = np.frombuffer(
                img_buf[:size],
                dtype=dtype).reshape((height, width, channels))

        yield arr
