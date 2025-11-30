""" 비디오영상 객체 추적
https://supervision.roboflow.com/latest/trackers/
"""
"""07_.py에서 추가 및 수정함"""

import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2

#yolo 초기 설정
model = YOLO("../../data/yolov/yolov8n.pt")

# 모든 객체 클래스 아이디와 이름을 저장
CLASS_NAME_DICT = model.model.names
# print(CLASS_NAME_DICT)

# 차량의 클래스 아이디 리스트 생성
classes = [2,5,7] # 2: 차, 5: 버스, 7: 트럭

# 비디오의 너비, 높이, fps, 총 프레임을 알려줌
# print(sv.VideoInfo.from_video_path(VIDEO_PATH))
# 결과값 : VideoInfo(width=3840, height=2160, fps=24, total_frames=1474)

# 영상 초기 설정
VIDEO_PATH = "../../data/video/highway.mp4"
captrue = cv2.VideoCapture(VIDEO_PATH)

# 윈도우창 초기 설정
win_name = "sv"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # 영상 크기 문제 해결

# 박스 및 라벨 인스턴스 생성
box_annotator = sv.BoxAnnotator(thickness=2) # 박스라벨의 두께 지정
label_annotator = sv.LabelAnnotator()

# [추가] 객체 추적 인스턴스 생성
tracker = sv.ByteTrack(frame_rate=24) # frame_rate=24 : 영상의 프레임 속도를 받아감(기본값은 30)

def process_frame(frame):
    """비디오 영상의 프레임을 처리하는 함수"""
    results = model(frame, verbose=False)[0]

    detections = sv.Detections.from_ultralytics(results)  # 변환 작업을 통해 detections에 객체 정보 저장
    detections = detections[np.isin(detections.class_id, classes)]  # 내가 원하는 객체들만 저장(detections.class_id에서 classes인것만 가져오기)

    # [추가] 객체 추적 id를 detections 정보에 추가한다
    detections = tracker.update_with_detections(detections)
    # print(detections) # tracker_id가 추가된게 보임

    # [수정] tracker_id를 추가함 / 라벨링 텍스트 작업(객체의 클래스 id, 객체 인식률)
    labels = []
    for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id):
        label = f'{tracker_id} {CLASS_NAME_DICT[class_id]} {confidence:0.2f}'
        labels.append(label)

    annotated_frames = box_annotator.annotate(frame.copy(), detections) # frame.copy(): 프레임을 훼손하지 않고 복사본에서 진행
    annotated_frames = label_annotator.annotate(annotated_frames, detections, labels)

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