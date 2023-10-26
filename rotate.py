import cv2
import numpy as np
import imutils as imu
import math

#  đọc  và tiền xử lý template
sr1= cv2.imread("data/sample-for-tets.bmp")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template, (3,3), 0)
src = cv2.Canny(blurred, 100, 150)

# Rotate src and mask by the same angle 'i'
src = imu.rotate(src, 30)



cv2.imshow('src',src)
cv2.waitKey(0)
cv2.destroyAllWindows()
