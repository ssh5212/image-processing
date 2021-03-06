import numpy as np
import math
import cv2

def exp(knN):
    th = -2 * math.pi *knN # 푸리에 변환 각도값
    return complex(math.cos(th), math.sin(th))

def dft(g):
    N = len(g)
    dst = [sum(g[n] * exp(-k*n/N) for n in range(N)) for k in range(N)]
    return np.array(dst)

def idft(g):
    N = len(g)
    dst = [sum(g[n] * exp(-k*n/N) for n in range(N)) for k in range(N)]
    return np.array(dst)/N

# 로그함수를 통해 잘 보이도록 스팩트럼을 만듦 = 범위를 줄임(시각화)
def calc_spectrum(complex):
    if complex.ndim==2:
        dst = abs(complex)
    else:
        dst = cv2.magnitude(complex[:, :, 0], complex[:, :, 1]) # cv의 경우
    dst = cv2.log(dst + 1)
    cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
    return cv2.convertScaleAbs(dst)

# 셔플링, 센터를 기준으로 총 4개로 쪼개고 위치를 서로 바꿈
def fftshift(img):
    dst = np.zeros(img.shape, img.dtype)
    h, w = dst.shape[:2]
    cy, cx = h // 2, w // 2
    dst[h-cy:, w-cx:] = np.copy(img[0:cy, 0:cx])
    dst[0:cy, 0:cx] = np.copy(img[h-cy:, w-cx:])
    dst[0:cy, w-cx:] = np.copy(img[h-cy:, 0:cx])
    dst[h-cy:, 0:cx] = np.copy(img[0:cy, w-cx:])
    return dst

def FFT(image, mode=2):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = fftshift(dft)
    spectrum = calc_spectrum(dft)
    return dft, spectrum

def IFFT(dft, shape, mode=2):
    dft = fftshift(dft)
    img = cv2.idft(dft, flags=cv2.DFT_SCALE)[:,:,0]
    img = img[:shape[0], :shape[1]]
    return cv2.convertScaleAbs(img)

