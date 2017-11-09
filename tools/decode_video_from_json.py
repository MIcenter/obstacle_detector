#!/usr/bin/python3

import cv2
import numpy as np

import json
import sys


def create_ndarray_from_string(json_info, string):
    string = bytearray.fromhex(string)
    dtype = np.float32
    channels = 1
    if json_info["format"] == 'bgr24':
        dtype = np.uint8
        channels = 3
    height = json_info["height"]
    width = json_info["width"]

    arr = np.frombuffer(string, dtype=json_info["dtype"]).reshape(height, width, channels)
    return arr


def read_json_frame_from_stdin():
    json_info_string = input()#sys.stdin.read()#input()
    json_info = json.loads(json_info_string)
    frame_string = input()#sys.stdin.read(int(json_info["size"]) * 2 + 1)

    return create_ndarray_from_string(json_info, frame_string)


def json_reader(json_read_function):
    while True:
        try:
            line = json_read_function()
        except EOFError:
            break
        yield line
