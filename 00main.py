import cv2
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

# استفاده از CAP_V4L2 برای سازگاری بهتر در لینوکس
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)

# چک کردن باز شدن دوربین
if not cam.isOpened():
    print("Error: Could not open webcam.")
    # تلاش برای ایندکس بعدی اگر دوربین اصلی پیدا نشد
    cam = cv2.VideoCapture(1, cv2.CAP_V4L2)
    if not cam.isOpened():
        exit()

while True:
    ret, frame = cam.read()
    
    if not ret or frame is None:
        print("Warning: Could not read frame, retrying...")
        continue

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_name, color_value in color_dict.items():
        
        lowerLimit, upperLimit = get_limits(color=color_value)

        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

        countours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if countours:
            max_contour = max(countours, key=cv2.contourArea)

            if cv2.contourArea(max_contour) > 700:
                x, y, w, h = cv2.boundingRect(max_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_value, 5)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_value, 2)

    cv2.imshow("mask", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()