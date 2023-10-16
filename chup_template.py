import cv2
import numpy as np
import imutils as imu

# doc anh va template
cap = cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    flip = cv2.flip(frame,-1)
    cv2.imshow("dectect",frame)
    if cv2.waitKey(1) == ord("q"):  # độ trễ 1/1000s , nếu bấm q sẽ thoát
        cv2.imwrite('data/air_src.jpg', flip)
        break

cv2.imshow("flip",flip)
cv2.waitKey(0)
cv2. destroyAllWindows