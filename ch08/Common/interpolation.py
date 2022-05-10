import numpy as np

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