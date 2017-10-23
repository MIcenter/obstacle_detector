import cv2


def sum_maps_equal(maps):
    result = maps[0]

    for i in range(1, len(maps)):
        result = cv2.addWeighted(
            result, i / (i + 1),
            maps[i], 1 - i / (i + 1),
            0)

    return result
