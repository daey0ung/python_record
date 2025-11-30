""" 01에서 읽어온 예측값을 sv를 통해서 처리 """

import cv2
import supervision as sv
from ultralytics import YOLO

IMG_PATH= "../../data/img/cat.jpg"
model = YOLO("../../data/yolov/yolov8n.pt") # yolo 8 나노 버전을 사용(n>s>m>l>x : 우측으로 갈수록 인식률이 좋으나 느려짐)
image = cv2.imread(IMG_PATH)

# 이미지에 yolo 적용
results = model(image)[0]
print('yolo를 적용한 이미지의 타입: ',type(results))
print(results.boxes)

print('-----------------------------------------------------------')

# yolo가 적용된 이미지를 sv로 변환
detections = sv.Detections.from_ultralytics(results)
print('yolo를 적용한 이미지를 sv로 변환 후 타입: ',type(detections))
print(detections) # 객체들이 리스트로 묶어서 xyxy, 인식률, class_id, tracker_id, data 값을 나타냄

print('-----------------------------------------------------------')

# 모든 객체 클래스 아이디와 이름을 저장
# coco에서 탐지 가능한 객체들 출력
CLASS_NAME_DICT = model.model.names
print(CLASS_NAME_DICT)