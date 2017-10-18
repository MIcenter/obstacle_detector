import cv2
from collections import deque, namedtuple
from functools import partial

Processed_frame = namedtuple('Processed_frame', 'frame gray blur')


def process_frame(frame, blur_value):
    return Processed_frame(\
        frame = frame,\
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),\
        blur = cv2.blur(frame, (blur_value, blur_value)))
        

class Frame_queue:
    def __init__(self, frames=None, blur_value=3):
        self.__blur_value = blur_value

        if frames is None:
            frames = []
        self.__frames = deque(
            map(
                partial(process_frame, blur_value=blur_value), frames))

    def __len__(self):
        return len(self.__frames)

    def __getitem__(self, index):
        return self.__frames[index]

    def popleft(self):
        return self.__frames.popleft()

    def append(self, frame):
        return self.__frames.append(process_frame(frame, self.__blur_value))
