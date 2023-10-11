import cv2
import imutils
import numpy
cap = cv2.VideoCapture(0)
rotate_degree = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (200,200))
    if ret:
        img = imutils.rotate(frame,rotate_degree)
        cv2.imshow("cam",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif cv2.waitKey(1) & 0xFF == ord('A'):
        rotate_degree += 90
    elif cv2.waitKey(1) & 0xFF == ord('B'):
        rotate_degree -= 90
cv2.destroyAllWindows()
cap.release()