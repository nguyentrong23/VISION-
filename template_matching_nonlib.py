import cv2
import numpy as np
import imutils as imu
import  math

# doc anh va template
sr0 = cv2.imread("data/bluepill.jpg")
img_template = cv2.cvtColor(sr0,cv2.COLOR_BGR2GRAY)
direc = 0.1
# sr1= cv2.imread("data/data_shape.png")
# img_src = cv2.cvtColor(sr1,cv2.COLOR_BGR2GRAY)
size = sr0.shape[::]

# bộ lọc sobel
gradient_x = cv2.Sobel(img_template, cv2.CV_64F, 1, 0, ksize=3)
gradient_x = np.abs(gradient_x)

gradient_y = cv2.Sobel(img_template, cv2.CV_64F, 0, 1, ksize=3)
gradient_y = np.abs(gradient_y)


# tạo các ma trận
nmsEdges = np.zeros_like(gradient_x)
mag_matrix =np.zeros_like(gradient_x)
direc_matrix =np.zeros_like(gradient_x)


# tính toán độ lớn gradient và hướng
for i in range(0,size[0],1):
    for j in range(0,size[1],1):
        sdx= gradient_x[i][j]
        sdy = gradient_y[i][j]
        mag= math.sqrt(sdx*sdx+sdy*sdy)
        mag_matrix[i][j] = mag
        direc = math.atan2(sdy,sdx) * 180/math.pi

        if (( 0 < direc < 22.5) or (157.5< direc < 202.5)or (337.5< direc < 360)):
                direc = 0
        elif(( 22.5 < direc < 67.5)or(202.5 < direc < 247.5)):
             direc = 45

        elif( ( 67.5 < direc < 112.5)or(247.5 < direc < 292.5)):
             direc = 90
        elif((112.5 < direc < 157.5)or(292.5 < direc < 337.5)):
             direc = 135
        else:
            direc = 0
        direc_matrix[i][j] = direc

# chuẩn hóa độ lớn gradient về giá trị 0-255
mag_matrix = np.uint8( mag_matrix)
maxgradient = np.max(mag_matrix)




 # non maximum suppression

for i in range(0,size[0]-1,1):
    for j in range(0,size[1]-1,1):
        if (direc_matrix[i][j]==0):
            leftPixel = mag_matrix[i][j - 1]
            rightPixel = mag_matrix[i][j + 1]
        elif(direc_matrix[i][j]==45):
            leftPixel = mag_matrix[i-1][j+1]
            rightPixel = mag_matrix[i+1][j-1]
        elif (direc_matrix[i][j] == 90):
            leftPixel = mag_matrix[i-1][j]
            rightPixel = mag_matrix[i+1][j]
        else:
            leftPixel = mag_matrix[i-1][j-1]
            rightPixel = mag_matrix[i+1][j+1]
        if (mag_matrix[i][j]<leftPixel or mag_matrix[i][j]<rightPixel):
            nmsEdges[i][j] =0
        else:
            nmsEdges[i][j] = mag_matrix[i][j]/maxgradient*255

# phân ngưỡng threshold
ret, nmsEdges = cv2.threshold( nmsEdges, 100, 140, cv2.THRESH_BINARY)



nmsEdges = cv2.resize(nmsEdges,(100,200))
cv2.imshow('template_edge',nmsEdges)
cv2.waitKey(0)
cv2. destroyAllWindows