import cv2
import time

mirror = True
from util import get_limits

color_dict = {
    "red": [0, 0, 255],
    "green": [0, 255, 0],
    "blue": [255, 0, 0],
    "yellow": [0, 255, 255],
    "orange": [0, 165, 255],
    "purple": [255, 0, 255],
    "white": [255, 255, 255],
    "black": [0, 0, 0]
}

# Create capture and set resolution
cam = cv2.VideoCapture(0)
# Set capture resolution to 1920x1080 (1080p)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
prev_time = time.time()
fps = 0.0
# Create named window for proper focus and key handling
window_name = "mask"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
try:
    while True:

        ret, frame = cam.read()
        if not ret:
            # If a frame was not read, skip processing and continue reading from camera
            continue
        # mirror the frame horizontally so view is like a mirror
        if mirror:
            frame = cv2.flip(frame, 1)

        # compute smoothed FPS
        current_time = time.time()
        dt = current_time - prev_time
        if dt > 0:
            new_fps = 1.0 / dt
            # smoothing for readability
            fps = fps * 0.9 + new_fps * 0.1 if fps else new_fps
        prev_time = current_time

        # print(color_dict)

        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for color_name,color_value in color_dict.items():

            lowerLimit, upperLimit =  get_limits(color=color_value)

            mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)  
  
    

            countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if countours:
                max_contour = max(countours, key=cv2.contourArea)

                if cv2.contourArea(max_contour) > 700:
                    x, y, w, h = cv2.boundingRect(max_contour)
                    bbox = (x, y, x + w, y + h)
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color_value, 5)

                    cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_value, 2)

        # show final frame (overlay added below)

        # put resolution and fps on top-right of frame
        try:
            width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        except Exception:
            width, height = frame.shape[1], frame.shape[0]

        text1 = f"{width}x{height}"
        text2 = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        pad = 8

        (w1, h1), b1 = cv2.getTextSize(text1, font, font_scale, thickness)
        (w2, h2), b2 = cv2.getTextSize(text2, font, font_scale, thickness)
        rect_w = max(w1, w2) + pad * 2
        rect_h = h1 + h2 + pad * 3
        rect_x1 = frame.shape[1] - rect_w - pad
        rect_y1 = pad
        rect_x2 = frame.shape[1] - pad
        rect_y2 = rect_y1 + rect_h

        # background rectangle for readability
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        # text positions
        text1_x = rect_x1 + pad
        text1_y = rect_y1 + pad + h1
        text2_x = rect_x1 + pad
        text2_y = text1_y + pad + h2
        cv2.putText(frame, text1, (text1_x, text1_y), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, text2, (text2_x, text2_y), font, font_scale, (255, 255, 255), thickness)


        # show final frame with overlays
        cv2.imshow(window_name, frame)

        # process keys, handle quitting via 'q' or Esc
        key = cv2.waitKey(1)
        if key != -1:
            if (key & 0xFF) in (ord('q'), ord('Q'), 27):
                break
        # if the window has been closed by user, stop
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
except KeyboardInterrupt:
    # allow graceful stop with ctrl+c
    pass
finally:
    cam.release()
    cv2.destroyAllWindows()


