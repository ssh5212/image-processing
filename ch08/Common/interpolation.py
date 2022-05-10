import numpy as np, cv2

# 정방행렬 인덱스로 크기 변경
def scaling(img, size):
    dst = np.zeros(size[::-1], img.dtype)
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2]) # 비율 계산
    y = np.arange(0, img.shape[0], 1)
    x = np.arange(0, img.shape[1], 1)
    y, x = np.meshgrid(y, x)
    i, j = np.int32(y * ratioY), np.int32(x * ratioX) # 목적 영상 좌표
    dst[i, j] = img[y, x]
    return dst

def scaling_nearest(img, size):
    dst = np.zeros(size[::-1], img.dtype)
    ratioY, ratioX = np.divide(size[::-1], img.shape[:2]) # 비율 계산
    i = np.arange(0, size[1], 1)
    j = np.arange(0, size[0], 1)
    i, j = np.meshgrid(i, j)
    y, x = np.int32(i / ratioY), np.int32(j / ratioX)
    dst[i, j] = img[y, x]

    return dst

def bilinear_value(img, pt):
    x, y = np.int32(pt)
    if x >= img.shape[1]-1: x = x - 1
    if y >= img.shape[0]-1: y = y - 1

    P1, P2, P3, P4 = np.float32(img[y:y+2, x:x+2].flatten())
    
    alpha, beta = pt[1] - y, pt[0] - x
    M1 = P1 + alpha * (P3 - P1) # 1차 보간
    M2 = P2 + alpha * (P4 - P2) 
    P = M1 + beta * (M2 - M1) # 2차 보간
    return np.clip(P, 0, 255)
