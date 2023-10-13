import cv2
import imutils
import numpy as np
import imutils as imu

# doc anh va template
sr0 = cv2.imread("data/data_shape.png")
img_src = cv2.cvtColor(sr0,cv2.COLOR_BGR2GRAY)

sr1= cv2.imread("data/template.png")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)

# list method co the dung khi matching template eval(cv2.TM_CCOEFF),eval(cv2.TM_CCORR_NORMED),eval(cv2.TM_CCORR),eval(cv2.TM_SQDIFF),cv2.TM_SQDIFF_NORMED
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = img_template.shape[::]
# white_background = np.ones((h,w, 1), np.uint8) * 255
# img_template = imu.rotate(img_template,33)
# result = cv2.addWeighted(white_background, 0.5, img_template, 0.5, 0)
cv2.imshow("hinh dang",sr0)
cv2.waitKey(0)
cv2. destroyAllWindows