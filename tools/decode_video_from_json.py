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


def decode_stdin():
    print('start decoding')
    json_line = sys.stdin.buffer.read()

    json_end = json_line.index(0)
    json_str = json_line[:json_end].decode('utf-8')

    json_info = json.loads(json_str)
    print(
        json_info["img-size"],
        json_info["height"],
        json_info["width"],
        json_info["format"],
        json_str,
        len(json_str),
        len(json_line)
        )

    size = int(json_info["img-size"])
    height = int(json_info["height"])
    width = int(json_info["width"])
    format_prop = json_info["format"]

    dtype = None
    channels = 1

    if format_prop == 'gray8':
        dtype = np.uint8
    elif format_prop == 'bgr24':
        dtype=np.uint8
        channels = 3

    if channels == 1:
        arr = np.frombuffer(
            json_line[json_end + 1:json_end + 1 + size],
            dtype=dtype).reshape((height, width))
    elif channels == 3:
        arr = np.frombuffer(
            json_line[json_end + 1:json_end + 1 + size],
            dtype=dtype).reshape((height, width, channels))

    print(arr)
    cv2.imshow('out', arr)
    cv2.waitKey()
    exit()


decode_stdin()
