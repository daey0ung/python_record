"""
실제 길이로 변경된 좌표로 속도 구하기
모든 차량을 계산하지만 박스 내 차량만 출력한다.
박스 외의 차량들은 xy가 올바르게 적용되지 않아서 표시하지 않는다.
"""

import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
import time
from collections import defaultdict, deque

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

# FPS 초기 설정
t0= time.time() # 현재 시간을 반환(1970년 1월 1일 0시 0분 0초부터 경과시간을 초단위로 반환)
frames_in_sec = 0 # 1초당 프레임 수를 저장
fps_text = "FPS: --" # 표시용 FPS 문자열 초기값

# 박스, 객체 추적, 객체 경로 인스턴스 생성
box_annotator = sv.BoxAnnotator(thickness=2)
tracker = sv.ByteTrack(frame_rate=24)
trace_annotator = sv.TraceAnnotator(thickness=2,trace_length=60)

# polygon 초기 설정
polygon = np.array([
    [1533,1246],[2560,1251],[3748,1900],[366,1835]
]) # 영역 좌표 설정
zone = sv.PolygonZone(polygon=polygon) # polygon zone 생성
polygon_annotator = sv.PolygonZoneAnnotator(zone, # polygon 인스턴스 생성
                                       color=sv.Color.WHITE, thickness=6, text_thickness=6, text_scale=4)

# 실제 거리 변환(투시변환)
TARGET_WIDTH = 25
TARGET_HEIGHT = 50
TARGET = np.array(
    [
        [0,0],
        [TARGET_WIDTH-1,0],
        [TARGET_WIDTH-1,TARGET_HEIGHT-1],
        [0,TARGET_HEIGHT-1]
    ]
)

class ViewTransformer:
    """ 픽셀좌표값을 실제 길이로 변경하는 클래스 생성 / https://m.blog.naver.com/0gon/221495813099 참고 """
    def __init__(self, source, target):
        source = source.astype(np.float32) # 프레임의 픽셀 좌표값 타입 변환
        target = target.astype(np.float32) # 투시 변환 좌표값 타입 변환
        self.m  = cv2.getPerspectiveTransform(source, target) # 프레임의 픽셀 좌표값이 투시 변환 좌표값으로 변경됨

    def transform_points(self, points):
        """ 탐지 객체의 하단 중심점을 받아와 연산 가능한 차원배열 형태로 변경과 타입 변환을 한 후 변경된 좌표값에 대입하여 실제 거리 좌표를 얻게됨 """
        reshaped = points.reshape(-1, 1, 2).astype(np.float32) # 넘겨 받은 좌표값을 배열 형태 변경하고 타입 변환함
        transformed = cv2.perspectiveTransform(reshaped, self.m) # 좌표값이 조정된 좌표값에 맞게 변형됨
        return transformed

view_transformer = ViewTransformer(polygon, TARGET) # 좌표값을 변경하는 인스턴스 생성

# [추가] 비디오 정보 저장,속도계산용 좌표 저장
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH) # 비디오 정보 저장
coordinates = defaultdict(lambda: deque(maxlen=video_info.fps)) # 속도계산용 좌표 저장

def process_frame(frame):
    """비디오 영상의 프레임을 처리하는 함수"""

    global frames_in_sec, t0, fps_text # FPS 관련 변수 선언

    results = model(frame, verbose=False, conf=0.5)[0]

    # 객체 정보 전처리 작업
    detections = sv.Detections.from_ultralytics(results)  # 변환 작업을 통해 detections에 객체 정보 저장
    detections = detections[np.isin(detections.class_id, classes)]  # 내가 원하는 객체들만 저장(detections.class_id에서 classes인것만 가져오기)
    detections = tracker.update_with_detections(detections)

    # 탐지 객체들의 하단 중심점을 points에 저장
    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    # print(points)

    # 탐지 객체들의 하단 중심점 리스트값이 도로 실제 길이(x,y)로 변경되는 과정
    if points.size > 0:
        points = points.astype(np.float32) # 타입 변경

        points = view_transformer.transform_points(points) # 도로 길이 클래스를 호출하면 좌표값을 도로 실제 길이(x,y)에 맞게 변경
        # print(f'클래스 적용후 :{points}') # 내가 설정한 도로 실제 길이(x,y)로 변환

        points = points.reshape(-1, 2) # 클래스 적용 후 배열 정렬
        # print(f'클래스 적용 후 배열 정렬 :{points}')
    else:
        # 빈 배열로 유지 (N, 2)
        points = np.empty((0, 2), dtype=np.float32)

    # polygon zone 트리거 선언 및 프레임에 polygon zone 표시
    is_detections_in_zone = zone.trigger(detections)  # 트리거를 활성화하고 zone에 있는지 상태를 저장한다

    # 프레임에 객체 박스, 객체 경로 표시 / polygon zone 안에 있는 탐지 객체들만 표시(mask기능을 사용)
    annotated_frames = box_annotator.annotate(frame.copy(), detections[is_detections_in_zone])  # frame.copy(): 프레임을 훼손하지 않고 복사본에서 진행
    annotated_frames = trace_annotator.annotate(annotated_frames, detections[is_detections_in_zone])

    # 프레임에 polygon zone 표시
    annotated_frames = polygon_annotator.annotate(annotated_frames)

    # [추가] 거리 구해서 라벨에 담기
    labels = []
    for tracker_id, [_,y] in zip(detections.tracker_id, points):
        coordinates[tracker_id].append(y)
        if len(coordinates[tracker_id]) < video_info.fps/2:
            labels.append(f'#{tracker_id}')
        else:
            coordinate_start = coordinates[tracker_id][-1] # 마지막 y 좌표
            coordinate_end = coordinates[tracker_id][0] # 처음 y좌표
            distance = abs(coordinate_end - coordinate_start) # y좌표 거리를 구하는데 음수면 양수로
            time_1 = len(coordinates[tracker_id]) / video_info.fps # y값은 담은 만큼을 영상의 프레임으로 나눠서 시간 구하기
            speed = distance / time_1 *3.6 # 거리 = 속도*시간 / 이러면 m/s가 나오고 여기에 3.6을 곱하면 km/h가 됨
            labels.append(f'#{tracker_id} {int(speed)} km/h')


    # 라벨링 쓰기 작업 / polygon에 있는 탐지 객체들만 라벨링
    for box, label in zip(detections[is_detections_in_zone].xyxy, labels):
        # print(f"{box}: {label}")
        x1, y1, x2, y2 = box.astype(int)  # 박스의 좌표를 정수로 변환, opencv함수 사용할때 좌표가 정수여야 해서
        cv2.putText(
            annotated_frames, label, (x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1,
            color=(0, 255, 255), thickness=3
        )

    # 프레임에 FPS 표시
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