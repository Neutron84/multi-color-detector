import numpy as np
import cv2

def get_limits(color):
    c = np.uint8([[color]])
    hsv_c = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsv_c[0][0][0]


    if color[0] == 0 and color[1] == 0 and color[2] == 0:  # Black
        lower_Limit = np.array([0, 0, 0], dtype=np.uint8)
        upper_Limit = np.array([180, 255, 30], dtype=np.uint8)
    elif color[0] == 255 and color[1] == 255 and color[2] == 255:  # White
        lower_Limit = np.array([0, 0, 200], dtype=np.uint8)
        upper_Limit = np.array([180, 20, 255], dtype=np.uint8)
    else:
        if hue >= 165:
             lower_Limit = np.array([hue - 10, 100, 100], dtype=np.uint8)
             upper_Limit = np.array([180, 255, 255], dtype=np.uint8)
        elif hue <= 15:
            lower_Limit = np.array([0, 100, 100], dtype=np.uint8)
            upper_Limit = np.array([hue + 10, 255, 255], dtype=np.uint8)
        else:
            lower_Limit = np.array([hue - 10, 100, 100], dtype=np.uint8)
            upper_Limit = np.array([hue + 10, 255, 255], dtype=np.uint8)


    return lower_Limit, upper_Limit