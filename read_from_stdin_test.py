#!/usr/bin/python3

import cv2
from tools.decode_video_from_json import decode_stdin


def show_video_from_sequence(gen):
    frame_number = 1

    for frame in gen:
        print(frame_number)
        frame_number += 1

        cv2.imshow('out', frame)

        if cv2.waitKey(1) == ord('q'):
            break

print('start test')
json_frames = decode_stdin()

show_video_from_sequence(json_frames)
