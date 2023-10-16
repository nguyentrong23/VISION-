import cv2
import numpy as np
import imutils as imu

# doc anh va template
cap = cv2.VideoCapture(0)
sr1= cv2.imread("data/bluepill.jpg")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)

# list method co the dung khi matching template eval(cv2.TM_CCOEFF),eval(cv2.TM_CCORR_NORMED),eval(cv2.TM_CCORR),eval(cv2.TM_SQDIFF),cv2.TM_SQDIFF_NORMED
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = img_template.shape[::]
topleft = [0,0]
cv2.imshow("template",sr1)

while 1:
    threshold = 0.4
    ret, frame = cap.read()
    img_src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # resolve angel problem
    for i in range(0,181,1):
        template = imu.rotate(img_template,i)
        res = cv2.matchTemplate(img_src, template, method)
        # xác dịnh tọa độ và vẽ khung cho template trên ảnh
        minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
        if(maxval>=threshold):
            res_cop = res
            print(threshold,':',i)
            threshold = maxval
            topleft = maxloc

#     de ve hinh chu nhat thi can biet toa do 2 goc cheo
    bottomright= (topleft[0]+w,topleft[1]+h)
    cv2.rectangle(frame,topleft,bottomright,(0,255,255),1)

    cv2.imshow("dectect",frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()  # giải phóng camera
cv2.destroyAllWindows()
