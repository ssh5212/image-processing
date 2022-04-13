import numpy as np, cv2

def draw_histo(hist, shape=(200, 256)): # shape : 히스토그램 이미지 사이즈
    hist_img = np.full(shape, 255, np.uint8) # 히스토그램을 그릴 화면
    cv2.normalize(hist, hist, 0, shape[0], cv2.NORM_MINMAX) # 정규화, 최솟값이 0이고 최대값이 그래프 영상의 높이를 가지도록 조절
    # 예제에서는 -> 4764(최대)가 200이 되도록 하겠다.
    gap = hist_img.shape[1]/hist.shape[0] # 한 계급 너비

    # 빈도 값에 대한 막대 사각형을 그림
    for i, h in enumerate(hist):
        x = int(round(i * gap)) # 시작 좌표
        w = int(round(gap))
        cv2.rectangle(hist_img, (x, 0, w, int(h)), 0, cv2.FILLED)
    
    return cv2.flip(hist_img, 0) # 영상 상하 뒤집기 # 원래는 위에서부터 아래로 내려가는 히스토그램이 생성됨