import  cv2
import  numpy

img1 = cv2.imread("data/road.jpg")
img2 = cv2.imread("data/car.jpg")

#  xu ly anh 2 de tao mask cho anh

gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

_,mask = cv2.threshold(gray,250,255,cv2.THRESH_BINARY)
cv2.imshow("mask", mask)
mask_not = cv2.bitwise_not(mask)

mask_road = cv2.bitwise_and(img1,img1,mask=mask)
mask_car = cv2.bitwise_and(img2,img2,mask=mask_not)
result = cv2.add(mask_road,mask_car)

cv2.imshow("result", result)
cv2.imshow("road tach mask", mask_road)
cv2.imshow("car tach mask", mask_car)
cv2.waitKey(0)
cv2.destroyAllWindows()
