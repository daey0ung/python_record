""" model.track을 사용하여 값 확인하기 """

import cv2, time
from ultralytics import YOLO  # yolo 라이브러리를 사용하기 위해 import

# 영상 초기 설정
VIDEO_PATH = '../../data/video/highway.mp4'
captrue = cv2.VideoCapture(VIDEO_PATH)
win_name = 'yolo'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# yolo 초기 설정
model = YOLO("../../data/yolov/yolov8n.pt") # yolo 8 나노 버전을 사용(n>s>m>l>x : 우측으로 갈수록 인식률이 좋으나 느려짐)

while captrue.isOpened():
    ret, frame = captrue.read()

    # yolo 적용
    # id값을 고정되게 하는 파라미터가 뭐더라??????????????
    result = model.track(frame)[0] # track을 사용하여 객체 추적 기능을 이용(id값을 가지게됨)
    # print(result)
    print(result.boxes[0]) # img1, img2참고
    yolo_frame = result.plot() # 객체를 박스 및 라벨링하여 yolo_frame에 저장

    # 영상 표시
    cv2.imshow(win_name, yolo_frame)

    # 무한 반복문 탈출 방법
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 프로그램 종료
captrue.release()
cv2.destroyAllWindows()
