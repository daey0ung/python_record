"""
10_에서 시작
차량 역주행 유뮤 확인
https://supervision.roboflow.com/latest/detection/tools/line_zone/#supervision.detection.line_zone.LineZoneAnnotator.__init__
out을 역주행으로 판단하여 기록에 남긴다.
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2

#yolo 초기 설정
model = YOLO("../../data/yolov/yolov8n.pt")
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

# line zone 인스턴스 생성 및 라인 좌표 생성
LINE_START = sv.Point(1200,1380) # 시작점
LINE_END = sv.Point(2800,1380) # 도착점
line_zone = sv.LineZone(LINE_START, LINE_END)
line_zone_annotator = sv.LineZoneAnnotator(thickness=2,text_thickness=2,text_scale=1) # 위에서 정의한 line_zone을 시각적으로 화면에 표시해주는 도구

# [추가] 차량 역주행, 정상주행 트래커 아이디를 담을 리스트
reverse_ids = [] # 역주행 트래커 아이디
ok_ids = [] # 정상주행 트래커 아이디

def process_frame(frame):
    """비디오 영상의 프레임을 처리하는 함수"""
    global reverse_ids, ok_ids

    results = model(frame, verbose=False)[0] # 프레임 추론, cmd에 값 노출x

    # 객체 정보 전처리 작업
    detections = sv.Detections.from_ultralytics(results) # yolo로 추론된 데이터를 sv 형식으로 바꿈
    detections = detections[np.isin(detections.class_id, classes)] # 원하는 탐지객체들(차량)만 선별
    detections = tracker.update_with_detections(detections) # traker_id 추가

    # 라벨링 텍스트 작업(객체의 클래스 id, 객체 인식률)
    labels = []
    for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id):
        label = f'{tracker_id} {CLASS_NAME_DICT[class_id]} {confidence:0.2f}'
        labels.append(label)

    # 프레임에 객체 박스 처리
    annotated_frames = box_annotator.annotate(frame.copy(), detections)
    annotated_frames = line_zone_annotator.annotate(annotated_frames, line_counter=line_zone)

    # 라벨링 쓰기 작업
    for box, label in zip(detections.xyxy, labels):
        x1, y1, x2, y2 = box.astype(int)
        cv2.putText(
            annotated_frames, label, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
            color=(0, 255, 255), thickness=3
        )

    # [수정] in,out에대한 불 연산자 분리
    crossed_in, crossed_out = line_zone.trigger(detections)

    # [추가] 트래커의 아이디 리스트 생성 / crossed_in, crossed_out의 불연산자 순서는 detections의 순서와 같다
    ids = detections.tracker_id

    # [추가]
    for i in range(len(ids)): # 탐지객체 수만큼 반복
        tid = ids[i] # 리스트에서 값을 뽑아 tid(tracker_id 줄임)에 저장
        if crossed_in[i]: # 방향 확인
            print(crossed_in[i])
            print()
            ok_ids.append(int(tid))  # 정상 방향에 저장
            ok_ids.sort() # 번호 정렬
        elif crossed_out[i]: # 방향 확인
            print(crossed_out[i])
            print()
            reverse_ids.append(int(tid))  # 역주행에 저장
            reverse_ids.sort() # 번호 정렬

    print(ok_ids)
    print(reverse_ids)

    cv2.putText(annotated_frames, f"ok_ids = {ok_ids}", (50, 120),
                cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(annotated_frames, f"reverse_ids = {reverse_ids}", (50, 70),
                cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

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