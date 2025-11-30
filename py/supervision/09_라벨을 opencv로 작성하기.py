""" 비디오영상 객체 추적 정보를 opencv로 입력 , fps 추가"""
"""08_.py에서 추가 및 수정함"""

import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
import time

#yolo 초기 설정
model = YOLO("../../data/yolov/yolov8n.pt")

# 객체 탐지 초기 설정
CLASS_NAME_DICT = model.model.names
classes = [2,5,7]

# 영상 초기 설정
VIDEO_PATH = "../../data/video/highway.mp4"
captrue = cv2.VideoCapture(VIDEO_PATH)

# 윈도우창 초기 설정
win_name = "sv"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # 영상 크기 문제 해결

# [수정] 박스 및 객체 추적 인스턴스 생성 / 라벨 인스턴스 삭제
box_annotator = sv.BoxAnnotator(thickness=2)
tracker = sv.ByteTrack(frame_rate=24)

# [추가] FPS 초기 설정
t0= time.time() # 현재 시간을 반환(1970년 1월 1일 0시 0분 0초부터 경과시간을 초단위로 반환)
frames_in_sec = 0 # 1초당 프레임 수를 저장
fps_text = "FPS: --" # 표시용 FPS 문자열 초기값

def process_frame(frame):
    """비디오 영상의 프레임을 처리하는 함수"""
    global frames_in_sec, t0, fps_text # FPS 관련 변수 선언

    results = model(frame, verbose=False)[0]

    # 객체 정보 전처리 작업
    detections = sv.Detections.from_ultralytics(results)  # 변환 작업을 통해 detections에 객체 정보 저장
    detections = detections[np.isin(detections.class_id, classes)]  # 내가 원하는 객체들만 저장(detections.class_id에서 classes인것만 가져오기)
    detections = tracker.update_with_detections(detections)

    # 라벨링 텍스트 작업(객체의 클래스 id, 객체 인식률)
    labels = []
    for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id):
        label = f'{tracker_id} {CLASS_NAME_DICT[class_id]} {confidence:0.2f}'
        labels.append(label)

    # 프레임에 객체 박스 처리
    annotated_frames = box_annotator.annotate(frame.copy(), detections) # frame.copy(): 프레임을 훼손하지 않고 복사본에서 진행

    # [추가] 라벨링 쓰기 작업
    for box, label in zip(detections.xyxy, labels):
        x1, y1, x2, y2 = box.astype(int)  # 박스의 좌표를 정수로 변환, opencv함수 사용할때 좌표가 정수여야 해서
        cv2.putText(
            annotated_frames, label, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
            color=(0, 255, 255), thickness=3
        )

    # [추가] 프레임에 FPS 표시
    frames_in_sec += 1  # 1초당 프레임 수 증가
    t1 = time.time()  # 현재 시간
    if t1 - t0 >= 1.0:  # 1초가 지났다면
        fps = frames_in_sec / (t1 - t0)  # 초당 프레임 계산
        fps_text = f'FPS: {fps:.1f}'  # 초당 프레임 출력용 저장
        t0, frames_in_sec = t1, 0  # 현재 시간값을 지난 시간값에 저장, 1초당 프레임 수 초기화
    # opencv를 이용하여 프레임에 글자 표시하기
    cv2.putText(annotated_frames, fps_text, (30, 60), # frame: 글씨를 적을 프레임, fps_text: 적을 문자열, (10,24): 문자열의 시작위치(좌측 하단 모서리 좌표),
                cv2.FONT_HERSHEY_SIMPLEX, 2,  # cv2.FONT_HERSHEY_SIMPLEX: 글꼴 , 0.7: 글자 크기
                (0, 255, 0), 2, cv2.LINE_AA)  # (0,255,0): 글자 색깔, 2: 글자 두께, cv2.LINE_AA: 글자 선타입

    return annotated_frames

# 윈도우창 무한 반복문
while captrue.isOpened():
    ret, frame = captrue.read()
    if not ret:
        break
    processed_frame = process_frame(frame) # 프레임 처리 함수 선언
    cv2.imshow(win_name, processed_frame) # 프레임 표시

    # 무한 반복문 탈출
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

captrue.release()
cv2.destroyAllWindows()