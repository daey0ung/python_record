""" 이미지의 작은 객체까지 탐지하는 방법 """

import cv2
import supervision as sv
from ultralytics import YOLO

# 이미지 초기 설정
IMG_PATH= "../../data/video/highway.mp4"
cap = cv2.VideoCapture(IMG_PATH)
ret, frame = cap.read()

# 이미지 출력화면 설정
win_name = 'sv'
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)

# yolo와 sv로 이미지 추론
model = YOLO("../../data/yolov/yolov8n.pt") # yolo 8 나노 버전을 사용(n>s>m>l>x : 우측으로 갈수록 인식률이 좋으나 느려짐)
results = model(frame, imgsz=1280)[0] # imgsz를 값을 높여 작은 객체도 인식되게 함
detections = sv.Detections.from_ultralytics(results)

# 박스 및 라벨 클래스 선언
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 박스 및 라벨을 이미지에 쓰기
annotated_image = box_annotator.annotate(frame, detections)
annotated_image = label_annotator.annotate(annotated_image, detections)

# 이미지 출력 설정
cv2.imshow(win_name, annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
