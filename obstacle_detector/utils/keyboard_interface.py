import cv2

def handle_keyboard(screenshot_image=None):
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        return 1
    elif k == ord('p'):
        cv2.waitKey()
    elif k == ord('s'):
        if screenshot_image is not None:
            cv2.imwrite('screen.png', img)
    return 0
