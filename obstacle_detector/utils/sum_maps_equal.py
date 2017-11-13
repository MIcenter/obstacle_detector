import cv2


def sum_maps_equal(maps, weights=None):
    result = maps[0]
    if weights is None:
        weights = [1 for i in maps]

    previous_weight = weights[0]
    for i, weight in enumerate(weights[1:]):
        result = cv2.addWeighted(
            result, previous_weight / (weight + previous_weight),
            maps[i + 1], weight / (weight + previous_weight),
            0)
        previous_weight += weight

    return result
