import cv2

cap = cv2.VideoCapture('../../data/video/cat.mp4')

ret, frame = cap.read()
cap.release()

if not ret:
    print("영상을 불러오지 못했습니다.")
    exit()

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # 좌클릭
        print(f'좌클릭 좌표: x={x}, y={y}')

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Frame', mouse_callback)

cv2.imshow('Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
