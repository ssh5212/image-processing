import numpy as np, cv2

def differental(image, data1, data2):
    mask1 = np.array(data1, np.float32).reshape(3, 3)
    mask2 = np.array(data1, np.float32).reshape(3, 3)

    dst1 = cv2.filter2D(image, -1, mask1)
    dst2 = cv2.filter2D(image, -1, mask2)
    dst1, dst2 = np.abs(dst1), np.abs(dst2) # 절대값 계산 == 양수로 변경
    # dst = cv2.magnitude(dst1, dst2) # 두 행렬의 크기 계산 == 엣지 강도 계산
    dst = dst1 + dst2

    dst = cv2.convertScaleAbs(dst) # 절대값 및 형변환
    dst1 = cv2.convertScaleAbs(dst1) # 절대값 및 형변환
    dst2 = cv2.convertScaleAbs(dst2) # 절대값 및 형변환
    return dst, dst1, dst2

# 침식 연산 함수
def erode(img, mask=None):
    dst = np.zeros(img.shape, np.uint8)
    if mask is None: mask = np.ones((3, 3), np.uint8)
    ycenter, xcenter = np.divmod(mask.shape[:2], 2)[0] # 마스크 중심 좌표

    mcnt = cv2.countNonZero(mask)
    for i in range(ycenter, img.shape[0] - ycenter):
        for j in range(xcenter, img.shape[1] - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1 # 마스크 높이 범위
            x1, x2 = j - xcenter, j + xcenter + 1 # 마스크 너비 범위
            roi = img[y1:y2, x1:x2] # 마스크 영역
            temp = cv2.bitwise_and(roi, mask)
            cnt = cv2.countNonZero(temp) # 일치 원소 개수 계산
            dst[i, j] = 255 if (cnt == mcnt) else 0 # 출력 화소에 저장
    return dst

# 팽창 연산 함수
def dilate(img, mask):
    dst = np.zeros(img.shape, np.uint8)
    if mask is None: mask = np.ones((3, 3), np.uint8)
    ycenter, xcenter = np.divmod(mask.shape[:2], 2)[0] # 마스크 중심 좌표

    for i in range(ycenter, img.shape[0] - ycenter):
        for j in range(xcenter, img.shape[1] - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1 # 마스크 높이 범위
            x1, x2 = j - xcenter, j + xcenter + 1 # 마스크 너비 범위
            roi = img[y1:y2, x1:x2] # 마스크 영역
            temp = cv2.bitwise_and(roi, mask)
            cnt = cv2.countNonZero(temp) # 일치 원소 개수 계산
            dst[i, j] = 0 if (cnt == 0) else 255 # 출력 화소에 저장
    return dst