import numpy as np
import serial
import cv2
import time

webcam = cv2.VideoCapture('')
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=0.1)
global value
value = 0

def write_data(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.06)
    data = arduino.readline()
    return data

while (1):
    _, imageFrame = webcam.read()
    imageFrame = cv2.resize(imageFrame, (860, 620))
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    red_lower = np.array([20, 100, 100], np.uint8)
    red_upper = np.array([40, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    kernal = np.ones((5, 5), "uint8")

    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=red_mask)

    # Creating contour to track red color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(value == "12"):
            write_data("0")
            value = "0"
        if (area>=400) :
            value = write_data("11")
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(imageFrame, str(), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))

    cv2.imshow("Window", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break