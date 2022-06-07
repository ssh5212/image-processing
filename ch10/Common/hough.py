import numpy as np, cv2, math

# 직선 좌표들에 대해 극 좌표계로 변환하여 인덱스 구성
def accumulate(image, rho, theta): # rho = 거리간격, theta = 각도 간격
    h, w = image.shape[:2]
    rows, cols = (h+w) * 2 // rho, int(np.pi / theta) # 누적행렬 너비, 높이
    accumulate = np.zeros((rows, cols), np.int32) # 직선 누적 행렬

    sin_cos = [(np.sin(t*theta), np.cos(t*theta)) for t in range(cols)]
    pts = np.where(image > 0) # 0이 아닌 점의 위치 저장

    polars = np.dot(sin_cos, pts).T # 극좌표 계산
    polars = (polars / rho + rows / 2).astype('int')

    for row in polars:
        for t, r in enumerate(row):
            accumulate[r, t] += 1 # 극좌표 누적
    return accumulate


# 지역 최대값 선정
def masking(accumulate, h, w, thresh):
    rows, cols = accumulate.shape[:2]
    rcenter, tcenter = h//2, w//2
    dst = np.zeros(accumulate.shape, np.uint32)

    for y in range(0, rows, h): # h, w = 마스크 크기
        for x in range(0, cols, w):
            roi = accumulate[y:y+h, x:x+w]
            print('cv2.minMaxLoc(roi) :', cv2.minMaxLoc(roi))
            _, max, _, (x0, y0) = cv2.minMaxLoc(roi)
            dst[y+y0, x+x0] = max
    return dst


# 극좌표 선택 및 정렬
def select_lines(acc_dst, rho, theta, thresh):
    rows = acc_dst.shape[0]
    r, t = np.where(acc_dst > thresh) # 임계값 이상의 인덱스만 가져옴

    rhos = ((r - (rows / 2)) * rho) # 인덱스로 수직 거리 계산
    radians = t * theta # 인덱스로 각도 계산
    values = acc_dst[r, t] # 인덱스로 누적값 가져옴

    idx = np.argsort(values)[::-1] # 내림차순 정렬 인덱스
    lines = np.transpose([rhos, radians])
    lines = lines[idx, :] # 누적값 기준으로 극좌표 정렬

    return np.expand_dims(lines, axis=1) # 차원 증가 -> cv와 동일하게 만들기 위하여


def hough_lines(src, rho, theta, thresh):
    acc_mat = accumulate(src, rho, theta)
    acc_dst = masking(acc_mat, 7, 3, thresh)
    lines = select_lines(acc_dst, 7, 3, thresh)
    return lines

def draw_hough_lines(src, lines, nline):
    # dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    dst = src
    min_length = min(len(lines), nline)

    for i in range(min_length):
        rho, radian = lines[i, 0, 0:2]
        a, b = math.cos(radian), math.sin(radian)
        pt = (a * rho, b * rho)
        delta = (-1000 * b, 1000 * a)
        pt1 = np.add(pt, delta).astype('int')
        pt2 = np.subtract(pt, delta).astype('int')
        cv2.line(dst, tuple(pt1), tuple(pt2), (0, 255, 0), 2, cv2.LINE_AA)

    return dst