import cv2
import numpy as np
import imutils as imu

# doc anh va template
cap = cv2.VideoCapture(0)

sr1= cv2.imread("data/template_cam.png")
img_template = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img_template, (3,3), 0)
edges_template = cv2.Canny(img_template, 50, 100)



# list method co the dung khi matching template eval(cv2.TM_CCOEFF),eval(cv2.TM_CCORR_NORMED),eval(cv2.TM_CCORR),eval(cv2.TM_SQDIFF),cv2.TM_SQDIFF_NORMED
method = eval("cv2.TM_CCOEFF_NORMED")
h, w = img_template.shape[::]
threshold = 0.4
topleft = [0,0]
cv2.imshow("template",edges_template)
while 1:
    ret, frame = cap.read()
    img_src = cv2.flip(frame,-1)
    img_src = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_src, (3,3), 0)
    edges_src = cv2.Canny(blurred, 50, 100)
    # resolve angel problem
    for i in range(0,181,2):
        template = imu.rotate(edges_template,i)
        res = cv2.matchTemplate(edges_src, template, method)
        # xác dịnh tọa độ và vẽ khung cho template trên ảnh
        minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
        if(maxval>=threshold):
            print(threshold,':',i)
            threshold = maxval
            topleft = maxloc

#     de ve hinh chu nhat thi can biet toa do 2 goc cheo
    bottomright= (topleft[0]+w,topleft[1]+h)
    cv2.rectangle(img_src,topleft,bottomright,(0,255,255),1)

    cv2.imshow("dectect",img_src)
    if cv2.waitKey(1) == ord("q"):  # độ trễ 1/1000s , nếu bấm q sẽ thoát
        break
cap.release()  # giải phóng camera
cv2.destroyAllWindows()
