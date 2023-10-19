import cv2
import numpy as np
import imutils as imu
import math

src = cv2.imread("data/data_shape.png")
cv2.circle(src,(62,50), 50, (0,255,0),1)
cv2.imshow('src', src)
cv2.waitKey(0)
cv2.destroyAllWindows()