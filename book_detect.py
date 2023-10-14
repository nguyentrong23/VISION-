import cv2
import numpy as np

cap =  cv2.VideoCapture(0)
while 1:
    ret, frame = cap.read()
    # _, threshold = cv2.threshold(frame, 155, 255, cv2.THRESH_BINARY)

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mean_c = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 12)
    gaus = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 12)

    cv2.imshow("Img", frame)
    # cv2.imshow("Binary threshold", threshold)
    cv2.imshow("Mean C", mean_c)
    cv2.imshow("Gaussian", gaus)
    if cv2.waitKey(1) == ord("q"):  # độ trễ 1/1000s , nếu bấm q sẽ thoát
        break
cv2.destroyAllWindows()