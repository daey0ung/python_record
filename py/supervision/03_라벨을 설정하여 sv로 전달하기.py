""" 라벨을 내가 설정하여 이미지에 쓰기 """

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

# 박스 및 라벨 인스턴스 생성
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 라벨 설정
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence in zip(detections['class_name'], detections.confidence)
]

print(detections)

# 박스 및 라벨을 이미지에 쓰기
annotated_image = box_annotator.annotate(image, detections)
annotated_image = label_annotator.annotate(annotated_image, detections, labels)

# 이미지 출력 설정
cv2.imshow('sv', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()