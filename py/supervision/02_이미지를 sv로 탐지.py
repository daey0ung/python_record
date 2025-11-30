""" 이미지에 sv로 주석 처리 """

import cv2
import supervision as sv
from ultralytics import YOLO

# 이미지 초기 설정
IMG_PATH= "../../data/img/cat.jpg"
image = cv2.imread(IMG_PATH)

# yolo와 sv로 이미지 추론
model = YOLO("../../data/yolov/yolov8n.pt") # yolo 8 나노 버전을 사용(n>s>m>l>x : 우측으로 갈수록 인식률이 좋으나 느려짐)
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)
print(detections[0])
print(detections[1])

points = detections[0].get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)  # shape: (N, 2)
print(points.astype(int))

# 박스 및 라벨 인스턴스 생성
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 박스 및 라벨을 이미지에 쓰기
annotated_image = box_annotator.annotate(image, detections)
annotated_image = label_annotator.annotate(annotated_image, detections)

# 이미지 출력 설정
cv2.imshow('sv', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
