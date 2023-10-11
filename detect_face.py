import cv2
import numpy as np
import  imutils
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
h= 200
w= 100
center = (h/2,w/2)
M = cv2.getRotationMatrix2D(center,180,1)
temp_rotate=0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (h,w))
    img = imutils.rotate(frame, temp_rotate + 180)# xoay cach 2 dung thu vien
    img = cv2.flip(img,-1)
    #img = cv2.warpAffine(image, M, (h, w))  # xoay cach 1 khong dung thu vien imutils
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,1.1,4)
    if len(faces) == 0:
        cv2.imshow("face ", img)
        continue
    for (x,y,z,t) in faces:
         cv2.rectangle(img, (x, y), (x + z, y + t), (0, 0, 255), 2)
    cv2.imshow("face ", img)
    if cv2.waitKey() & 0xFF == ord('q'):
        break
    elif cv2.waitKey() & 0xFF == ord('A'):
        temp_rotate += 90
    elif cv2.waitKey() & 0xFF == ord('B'):
        temp_rotate -= 90
cv2.destroyAllWindows()
cap.release()