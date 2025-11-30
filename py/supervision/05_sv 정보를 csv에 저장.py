""" sv로 객체 정보를 csv에 저장 """

import supervision as sv
from ultralytics import YOLO

model = YOLO("../../data/yolov/yolov8n.pt")
frames_generator = sv.get_video_frames_generator("../../data/video/highway.mp4")

with sv.CSVSink("csv/1.csv") as sink:
    for frame in frames_generator:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result) # sv로 탐지한 객체 정보들을
        sink.append(detections, {}) # csv에 저장

    # 사용자가 필드를 추가하는 방법
    # for frame_index, frame in enumerate(frames_generator):
    #     result = model(frame)[0]
    #     detections = sv.Detections.from_ultralytics(result) # sv로 탐지한 객체 정보들을
    #     sink.append(detections, {"frame_index": frame_index}) # csv에 저장