import numpy as np, cv2

def differental(image, data1, data2):
    mask1 = np.array(data1, np.float32).reshape(3, 3)
    mask2 = np.array(data1, np.float32).reshape(3, 3)

    dst1 = filter(image, mask1)
    dst2 = filter(image, mask2)
    dst = cv2.magnitude(dst1, dst2) # 두 행렬의 크기 계산 == 엣지 강도 계산

    dst = cv2.convertScaleAbs(dst) # 절대값 및 형변환
    dst1 = cv2.convertScaleAbs(dst1) # 절대값 및 형변환
    dst2 = cv2.convertScaleAbs(dst2) # 절대값 및 형변환
    return dst, dst1, dst2
