""" 고속도로 영상에 나오는 차량들을 탐지한다. """

import cv2, time
from ultralytics import YOLO  # yolo 라이브러리를 사용하기 위해 import

# 영상 초기 설정
VIDEO_PATH = '../../data/video/highway.mp4'
captrue = cv2.VideoCapture(VIDEO_PATH)
win_name = 'yolo'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# yolo 초기 설정
model = YOLO("../../data/yolov/yolov8n.pt") # yolo 8 나노 버전을 사용(n>s>m>l>x : 우측으로 갈수록 인식률이 좋으나 느려짐)

# 영상 시작
while captrue.isOpened():
    # 영상 읽기
    ret, frame = captrue.read()

    # yolo 적용
    # model(frame) / model.predict(frame) 은 서로 같아서 아무거나 사용
    result = model(frame)[0] # 프레임에 대해 yolo 추론을 실행 후 결과값을 result에 저장
    print(result) # img1, img2참고
    yolo_frame = result.plot() # 객체를 박스 및 라벨링하여 yolo_frame에 저장

    # 영상 표시
    cv2.imshow(win_name, yolo_frame)

    # 무한 반복문 탈출 방법
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 프로그램 종료
captrue.release()
cv2.destroyAllWindows()
