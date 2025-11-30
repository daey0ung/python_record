""" yolo segmentation 작업중인 이미지에 mask 처리 """

import cv2
import supervision as sv
from ultralytics import YOLO

# 이미지 초기 설정
IMG_PATH= "../../data/img/cat.jpg"
image = cv2.imread(IMG_PATH)

# yolo와 sv로 이미지 추론
model = YOLO("../../data/yolov/yolov8n-seg.pt") # seg로 처리해야 마스크 처리됨
results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

# 마스크 및 라벨 인스턴스 생성
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER_OF_MASS)

# mask 처리시 mask값에 불연산자가 생긴다
print(detections)

# 박스 및 라벨을 이미지에 쓰기
annotated_image = mask_annotator.annotate(image, detections)
annotated_image = label_annotator.annotate(annotated_image, detections)

# 이미지 출력 설정
cv2.imshow('sv', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()