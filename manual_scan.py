"""
폰 카메라로 찍은 사진, pc에서 손으로 스캔하기
코드 참고: https://github.com/BaekKyunShin/OpenCV_Project_Python/blob/master/05.geometric_transform/perspective_scan.py

# 변수 설정
- 스캔을 원하는 이미지들이 들어가있는 경로를 'PATH_NAME'에 작성
  (ex. C:/Path/To/File/*.png)
- 이미지가 너무 해상도가 높을 경우 'IMAGE_SIZE_REDUCTION_RATIO'로 가로세로 크기를 N배 줄이겠다고 정의
# 사용법
- 스캔화를 원하는 부분 꼭지점 4개 찍기
- 4개 점 찍은 후 나오는 화면으로 스캔 결과 확인
- z, c 키로 앞뒤 이미지 이동 가능
- 슬라이더 클릭으로 이미지 이동 가능
"""

import cv2
import numpy as np
from glob import glob
import os

WIN_NAME = "scanning"
ROOT_PATH = 'C:/Users/J/Downloads/test'
PATH_NAME = f'{ROOT_PATH}/*/*.jpg'
IMAGE_SIZE_REDUCTION_RATIO = 0.25
IMAGE_SIZE_RECOVER_RATIO = int(1/IMAGE_SIZE_REDUCTION_RATIO)

global img, win_img, file_name, file_extension, parent_dir, rows, cols, draw, pts_cnt, pts

def is_in_range(idx): # 인덱싱 에러 확인용 함수
    return 0 <= idx <= len(path_list)

def set_variables(idx): # 변수 초기화 함수
    global img, win_img, file_name, file_extension, parent_dir, rows, cols, draw, pts_cnt, pts
    
    img = cv2.imread(path_list[idx])
    file_name, file_extension = os.path.basename(path_list[idx]).split('.')
    parent_dir = os.path.dirname(path_list[idx])
    win_img = cv2.resize(img, (0, 0), fx=IMAGE_SIZE_REDUCTION_RATIO, fy=IMAGE_SIZE_REDUCTION_RATIO)
    rows, cols = win_img.shape[:2]
    draw = win_img.copy()
    pts_cnt = 0
    pts = np.zeros((4,2), dtype=np.float32)

def onChange(pos): # 슬라이더 변화 감지 콜백 함수 구현
    global win_img
    idx = cv2.getTrackbarPos("idx", WIN_NAME)
    if not is_in_range(idx):
        return
    set_variables(idx)
    cv2.imshow(WIN_NAME, win_img)

def onMouse(event, x, y, flags, param):  # 마우스 이벤트 콜백 함수 구현
    global img, win_img, file_name, file_extension, parent_dir, rows, cols, draw, pts_cnt, pts
    if event == cv2.EVENT_LBUTTONDOWN:  
        cv2.circle(draw, (x,y), 10, (0,255,0), -1) # 좌표에 초록색 동그라미 표시
        cv2.imshow(WIN_NAME, draw)

        pts[pts_cnt] = [x,y]            # 마우스 좌표 저장
        pts_cnt+=1
        if pts_cnt == 4:                       # 좌표가 4개 수집됨 
            # 좌표 4개 중 상하좌우 찾기
            sm = pts.sum(axis=1)                 # 4쌍의 좌표 각각 x+y 계산
            diff = np.diff(pts, axis = 1)       # 4쌍의 좌표 각각 x-y 계산

            topLeft = pts[np.argmin(sm)]         # x+y가 가장 값이 좌상단 좌표
            bottomRight = pts[np.argmax(sm)]     # x+y가 가장 큰 값이 우하단 좌표
            topRight = pts[np.argmin(diff)]     # x-y가 가장 작은 것이 우상단 좌표
            bottomLeft = pts[np.argmax(diff)]   # x-y가 가장 큰 값이 좌하단 좌표

            # 변환 전 4개 좌표 
            pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])

            # 변환 후 영상에 사용할 서류의 폭과 높이 계산
            w1 = abs(bottomRight[0] - bottomLeft[0])    # 상단 좌우 좌표간의 거리
            w2 = abs(topRight[0] - topLeft[0])          # 하단 좌우 좌표간의 거리
            h1 = abs(topRight[1] - bottomRight[1])      # 우측 상하 좌표간의 거리
            h2 = abs(topLeft[1] - bottomLeft[1])        # 좌측 상하 좌표간의 거리
            width = int(max([w1, w2]))                       # 두 좌우 거리간의 최대값이 폭
            height = int(max([h1, h2]))                      # 두 상하 거리간의 최대값이 높이

            # 변환 후 4개 좌표
            pts2 = np.float32([[0,0], [width-1,0], 
                                [width-1,height-1], [0,height-1]])
            # 변환 행렬 계산 
            mtrx = cv2.getPerspectiveTransform(pts1*IMAGE_SIZE_RECOVER_RATIO, pts2*IMAGE_SIZE_RECOVER_RATIO)
            # 원근 변환 적용
            result = cv2.warpPerspective(img, mtrx, (width*IMAGE_SIZE_RECOVER_RATIO, height*IMAGE_SIZE_RECOVER_RATIO))
            # 처리한 이미지 정리용 폴더 생성 후 저장
            if not os.path.isdir(f'{parent_dir}/scanned'):
                os.mkdir(f'{parent_dir}/scanned')
            cv2.imwrite(f'{parent_dir}/scanned/{file_name}_scanned.{file_extension}', result)
            # 처리된 이미지 확인창 생성
            win_result = cv2.resize(result, (0, 0), fx=IMAGE_SIZE_REDUCTION_RATIO, fy=IMAGE_SIZE_REDUCTION_RATIO)
            cv2.imshow('scanned', win_result)
            print(f"{parent_dir}/scanned/{file_name}_scanned.{file_extension} saved!")

path_list = sorted(list(glob(PATH_NAME)))

set_variables(0)

cv2.imshow(WIN_NAME, win_img)
cv2.createTrackbar("idx", WIN_NAME, 0, len(path_list)-1, onChange) # 이미지 인덱싱용 슬라이더 GUI 윈도우에 등록
cv2.setMouseCallback(WIN_NAME, onMouse)    # 마우스 콜백 함수를 GUI 윈도우에 등록

while True:
    if cv2.waitKey() == ord('z'):
        idx = cv2.getTrackbarPos("idx", WIN_NAME)
        if is_in_range(idx):
            cv2.setTrackbarPos("idx", WIN_NAME, idx-1)

    elif cv2.waitKey() == ord('c'):
        idx = cv2.getTrackbarPos("idx", WIN_NAME)
        if is_in_range(idx):
            cv2.setTrackbarPos("idx", WIN_NAME, idx+1)

    elif cv2.waitKey() == ord('q'):
        break


cv2.destroyAllWindows()
