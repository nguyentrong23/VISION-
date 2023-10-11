import cv2


image = cv2.VideoCapture(0)
while (True):
    ret,frame = image.read()
    frame= cv2.resize(frame,(680,420))
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
image.release()
cv2.destroyAllWindows()
