""" 비디오영상에 라인을 생성하여 차량의 in,out을 측정
https://supervision.roboflow.com/latest/detection/tools/line_zone/#supervision.detection.line_zone.LineZoneAnnotator.__init__

"""
"""역주행 코드"""

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

# [추가] line zone 인스턴스 생성 및 라인 좌표 생성
LINE_START = sv.Point(1200,1380) # 시작점
LINE_END = sv.Point(2800,1380) # 도착점
line_zone = sv.LineZone(LINE_START, LINE_END)
line_zone_annotator = sv.LineZoneAnnotator(thickness=2,text_thickness=2,text_scale=1) # 위에서 정의한 line_zone을 시각적으로 화면에 표시해주는 도구

def process_frame(frame):
    """비디오 영상의 프레임을 처리하는 함수"""
    results = model(frame, verbose=False)[0]

    # 객체 정보 전처리 작업
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[np.isin(detections.class_id, classes)]
    detections = tracker.update_with_detections(detections)

    # 라벨링 텍스트 작업(객체의 클래스 id, 객체 인식률)
    labels = []
    for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id):
        label = f'{tracker_id} {CLASS_NAME_DICT[class_id]} {confidence:0.2f}'
        labels.append(label)

    # 프레임에 객체 박스 처리
    annotated_frames = box_annotator.annotate(frame.copy(), detections)

    # 라벨링 쓰기 작업
    for box, label in zip(detections.xyxy, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.putText(
            annotated_frames, label, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
            color=(0, 255, 255), thickness=3
        )

    # [추가] line zone 트리커 활성화 및 프레임에 line zone 표시
    info =line_zone.trigger(detections) # 선을 넘나드는 객체 카운트
    print(info)
    annotated_frames = line_zone_annotator.annotate(annotated_frames, line_counter=line_zone)

    return annotated_frames

while captrue.isOpened():
    ret, frame = captrue.read()
    if not ret:
        break
    processed_frame = process_frame(frame)
    cv2.imshow(win_name, processed_frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

captrue.release()
cv2.destroyAllWindows()