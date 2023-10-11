import cv2
import  numpy
import  imutils

img = cv2.imread("data/data_shape.png")
# b1 convert ve anh xam
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(grey,threshold1=100,threshold2=200)
cv2.imshow("bien",edges)
cv2.waitKey()
if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()


