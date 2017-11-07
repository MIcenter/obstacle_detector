import cv2


def sum_maps_equal(maps, weights=None):
    result = maps[0]
    if weights is None:
        weights = [1 for i in maps]

    previous_weight = weights[0]
    for i in weights[1:]:
        result = cv2.addWeighted(
            result, previous_weight / (i + previous_weight),
            maps[i], i / (i + previous_weight),
            0)
        i += previous_weight

    return result
