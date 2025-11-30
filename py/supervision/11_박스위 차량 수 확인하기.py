""" 비디오 영상에 영역을 설정하여 영역 위의 객체 숫자 탐지
https://supervision.roboflow.com/latest/detection/tools/polygon_zone/
 """
"""주정차 코드 및 차로별 차량수 확인 코드"""

import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2

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

# 박스 및 객체 추적 인스턴스 생성
box_annotator = sv.BoxAnnotator(thickness=2)
tracker = sv.ByteTrack(frame_rate=24)

# [추가] polygon 초기 설정
polygon = np.array([
    [2145, 1283],
    [2270, 1272],
    [2652, 1792],
    [2365, 1820]
]) # 영역 좌표 설정
zone = sv.PolygonZone(polygon=polygon) # polygon zone 생성
polygon_annotator = sv.PolygonZoneAnnotator(zone, # polygon 인스턴스 생성
                                       color=sv.Color.WHITE, thickness=6, text_thickness=6, text_scale=4)

def process_frame(frame):
    """비디오 영상의 프레임을 처리하는 함수"""
    results = model(frame, verbose=False, conf=0.5)[0]

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

    # 라벨링 쓰기 작업
    for box, label in zip(detections.xyxy, labels):
        x1, y1, x2, y2 = box.astype(int)  # 박스의 좌표를 정수로 변환, opencv함수 사용할때 좌표가 정수여야 해서
        cv2.putText(
            annotated_frames, label, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
            color=(0, 255, 255), thickness=3
        )

    # [추가] polygon zone 트리거 선언 및 프레임에 polygon zone 표시
    is_detections_in_zone= zone.trigger(detections) # 트리거를 활성화하고 zone에 있는지 상태를 저장한다
    # print(is_detections_in_zone) # 불 연산자로 나옴
    annotated_frames = polygon_annotator.annotate(annotated_frames)

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