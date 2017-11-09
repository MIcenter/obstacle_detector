#!/usr/bin/python3

import cv2
from tools.decode_video_from_json import *


def show_video_from_sequence(gen):
    for frame in gen:
#        print(type(frame.shape))
        cv2.imshow('out', frame)
        cv2.waitKey(1)

json_frames = json_reader(read_json_frame_from_stdin)

show_video_from_sequence(json_frames)
