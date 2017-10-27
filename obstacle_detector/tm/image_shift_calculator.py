import cv2


def find_shift_value(img, old_img, coords):
    x1, y1, x2, y2 = coords
    old_template = old_img[y1:y2, x1:x2].copy()

    res = cv2.matchTemplate(img, old_template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc

    t_height, t_width = old_template.shape[:2]

    return top_left[0] - coords[0], top_left[1] - coords[1]


def find_template_in_img(template, img):
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc

    t_height, t_width = template.shape

    bottom_right = (top_left[0] + t_width, top_left[1] + t_height)
    (x1, y1), (x2, y2) = top_left, bottom_right

    return x1, y1, x2, y2
