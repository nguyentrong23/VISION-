import cv2
import  imutils as imu
import numpy as np
# doc anh va template
sr0 = cv2.imread("data/data_shape.png")
img_src = cv2.cvtColor(sr0,cv2.COLOR_BGR2GRAY)

sr1= cv2.imread("data/template.png")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
# list method co the dung khi matching template eval(cv2.TM_CCOEFF),eval(cv2.TM_CCORR_NORMED),eval(cv2.TM_CCORR),eval(cv2.TM_SQDIFF),cv2.TM_SQDIFF_NORMED
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = img_template.shape[::]

res = cv2.matchTemplate(img_src, img_template, method)
print(cv2.minMaxLoc(res))

# xác dịnh tọa độ và vẽ khung cho template trên ảnh
minval,maxval,minloc,maxloc = cv2.minMaxLoc(res)

if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    topleft = minloc
else:
    topleft = maxloc
#     de ve hinh chu nhat thi can biet toa do 2 goc cheo
bottomright= (topleft[0]+w,topleft[1]+h)
cv2.rectangle(sr0,topleft,bottomright,(0,255,255),1)

#  su dung phung phap counter contours de xac dinh shape can tim
# sr = cv.imread("data/data_shape.png",1)
# img = cv.cvtColor(sr,cv.COLOR_BGR2GRAY)
#
# _, threshold =cv.threshold(img,80,180, cv.THRESH_BINARY)
# contours,_ = cv.findContours(threshold,cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
# for cnt in contours:
#     approx=cv.approxPolyDP(cnt, 0.01*cv.arcLength(cnt, True),True)
#     cv.drawContours(sr,[approx],-1,(0,250,0),1)
#     x = approx.ravel()[0]
#     y=  approx.ravel()[1]
#     if len(approx)==5:
#         cv.putText(sr,"ngugiac",(x,y),cv.FONT_HERSHEY_COMPLEX,1,(0,255,255),1,cv.LINE_AA)
#         cv.drawContours(sr,[approx],-1,(255,250,255),2)
#     if len(approx)==10:
#         cv.putText(sr,"sao 5 canh",(x,y),cv.FONT_HERSHEY_COMPLEX,1,(0,255,255),1,cv.LINE_AA)
#         cv.drawContours(sr,[approx],-1,(255,250,0),2)
cv2.imshow("hinh dang",sr0)
cv2.waitKey(0)
cv2. destroyAllWindows
