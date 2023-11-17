import cv2
import numpy as np
import serial
import cv2
import time


webcam = cv2.VideoCapture(0)
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=0.1)
global value
value = 0
def write_data(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.06)
    data = arduino.readline()
    return data

image = cv2.imread( )
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#chuyen ve anh xam
ret, thresh = cv2.threshold(gray, 230,255,cv2.THRESH_BINARY)  #dua ve anh nhi phan tu anh xam theo nguong
# cv2.imshow("B", thresh)  # hien thi test anh nhi phan

contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # tim contours tu anh nhi phan
#
AREA = image.shape[0]*image.shape[1]/150 # kich thuoc toi thieu de duoc ve contours
dem = 0
for cnt in contours[:-1]:
    if cv2.contourArea(cnt) >= AREA:
        dem+=1
        cv2.drawContours(image, [cnt], -1, (0,255,0), 2, cv2.LINE_AA)
# cv2.drawContours(image, contours,-1, (0,255,0),2)# vẽ lại ảnh contour vào ảnh gốc 
print(dem)
cv2.imshow("A", image)
cv2.waitKey()
cv2.destroyAllWindows()