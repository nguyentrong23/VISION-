import cv2
import numpy as np
import imutils as imu
import math
modelDefined = False
noOfCordinates = 0
cordinates = None
edgeMagnitude = None
edgeDerivativeX = None
edgeDerivativeY = None
modelHeight = None
modelWidth = None
centerOfGravity = None

def create_double_matrix(height, width):
    return np.zeros((height, width), dtype=float)

# Function to release a 2D matrix
def release_double_matrix(matrix):
    return None

#
# def find_geo_match_model(srcarr,template,lastMaxScores):
#     gradient_srcx, gradient_srcy = calculate_gradients(srcarr)
#     gradient_x, gradient_y = calculate_gradients(template)
#     mag_src,_ =compute_magnitude_direction(gradient_srcx,gradient_srcy)
#     mag_tem,_ = compute_magnitude_direction(gradient_x,gradient_y)
#     size_temp= mag_tem.shape
#     size_src = mag_src.shape
#     result = np.zeros_like(mag_src)
#     for i in range(1, size_src[0]):
#         for j in range(1, size_src[1]):
#             point =0
#             for n in range(1,size_temp[0]):
#                 for m in range(1, size_temp[1]):
#                     point = point + (gradient_x[n][m]*gradient_srcx[i][j]+gradient_y[n][m]*gradient_srcy[i][j])/(mag_src[i][j] *mag_tem[n][m])
#                     # point =point +((gradient_x[n][m]*gradient_srcx[i][j]+gradient_y[n][m]*gradient_srcy[i][j])/(mag_src[i][j] *mag_tem[n][m]))
#             result[i][j] = point/(size_temp[0]*size_temp[1])
#             print(result[i][j])
#     return  result
#     for i in range(1, size_src[0]):
#         for j in range(1, size_src[1]):
#             print(mag_src[i][j])
#     return 1

def calculate_gradients(img):
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_x = np.abs(gradient_x)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_y = np.abs(gradient_y)
    return gradient_x, gradient_y


def compute_magnitude_direction(gradient_x, gradient_y):
    mag_matrix = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    direc_matrix = np.degrees(np.arctan2(gradient_y, gradient_x))
    # Convert angles to 0-180 degrees
    direc_matrix[direc_matrix < 0] += 180
    return mag_matrix, direc_matrix


def non_maximum_suppression(mag_matrix, direc_matrix):
    size = mag_matrix.shape
    nmsEdges = np.zeros_like(mag_matrix)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            if (direc_matrix[i][j] == 0):
                leftPixel = mag_matrix[i][j - 1]
                rightPixel = mag_matrix[i][j + 1]
            elif (direc_matrix[i][j] == 45):
                leftPixel = mag_matrix[i - 1][j + 1]
                rightPixel = mag_matrix[i + 1][j - 1]
            elif (direc_matrix[i][j] == 90):
                leftPixel = mag_matrix[i - 1][j]
                rightPixel = mag_matrix[i + 1][j]
            else:
                leftPixel = mag_matrix[i - 1][j - 1]
                rightPixel = mag_matrix[i + 1][j + 1]
            if (mag_matrix[i][j] < leftPixel or mag_matrix[i][j] < rightPixel):
                nmsEdges[i][j] = 0
            else:
                nmsEdges[i][j] = mag_matrix[i][j]

    return nmsEdges


def apply_threshold(nmsEdges, low_threshold, high_threshold):
    _, nmsEdges = cv2.threshold(nmsEdges, low_threshold, high_threshold, cv2.THRESH_BINARY)
    return nmsEdges

def CreateGeoMatchModel(img_template):
    gradient_x, gradient_y = calculate_gradients(img_template)
    mag_matrix, direc_matrix = compute_magnitude_direction(gradient_x, gradient_y)
    nmsEdges = non_maximum_suppression(mag_matrix, direc_matrix)
    nmsEdges = apply_threshold(nmsEdges, 90, 110)
    return nmsEdges


def main():
    global modelDefined, noOfCordinates, cordinates, edgeMagnitude, edgeDerivativeX, edgeDerivativeY, modelHeight, modelWidth, centerOfGravity
    threshold = 0.0
    sr0 = cv2.imread("data/sample_for_template.bmp")
    sr0 = cv2.cvtColor(sr0, cv2.COLOR_BGR2GRAY)
    sourc = CreateGeoMatchModel(sr0)
    method = eval("cv2.TM_CCOEFF_NORMED")
    sr1 = cv2.imread("data/sample-2-4.bmp")
    sr1 = cv2.cvtColor(sr1, cv2.COLOR_BGR2GRAY)
    template = CreateGeoMatchModel(sr1)
    cv2.imshow('source',sourc)
    cv2.imshow('template',template)
    for i in range(0, 181, 1):
        src_rotate = imu.rotate(sourc, i)
        res = cv2.matchTemplate(src_rotate, template, method)
        # xác dịnh tọa độ và vẽ khung cho template trên ảnh
        minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
        if (maxval >= threshold):
            print(threshold, ':', i)
            best = i
            threshold = maxval
            topleft = maxloc

    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
