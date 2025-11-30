""" 동영상 출력하기 """
# 동영상 정보 : https://pixabay.com/ko/videos/%ea%b3%a0%ec%96%91%ec%9d%b4-%ea%b3%a0%ec%96%91%ec%9d%b4-%eb%88%88-143890/
# 동영상은 data/video 폴더에 있고 상대 경로를 사용해서 가져온다.

import cv2

# 비디오 출력 클래스(cv2.VideoCapture)를 통해 내장 카메라 또는 외장 카메라에서 정보를 받아온다.
# cv2.VideoCapture(index)로 카메라의 장치 번호(ID)와 연결합니다. index는 카메라의 장치 번호를 의미합니다.
# 노트북 카메라의 장치 번호는 0 / 카메라를 추가적으로 연결하여 외장 카메라를 사용하는 경우, 장치 번호가 1~n까지 순차적으로 할당됩니다.
# capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture('../../data/video/cat.mp4')

# 카메라 속성 설정 메서드(capture.set)로 카메라의 속성을 설정합니다.
# capture.set(propid, value)로 카메라의 속성(propid)과 값(value)을 설정할 수 있습니다.
# propid은 변경하려는 카메라 설정을 의미합니다.
# value은 변경하려는 카메라 설정의 속성값을 의미합니다.
# 예제에서는 카메라의 너비를 640, 높이를 480으로 변경합니다.
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while capture.isOpened(): # q가 입력될 때 while문을 종료합니다.
    """반복문(While)을 활용하여 카메라에서 프레임을 지속적으로 받아옵니다."""
    # capture.read()를 이용하여 카메라의 상태값 및 프레임을 가져온다.
    # ret= True(카메라 정삭 작동 시), False(카메라 미 작동 시)
    # frame= 현재 시점의 프레임이 저장됨
    ret, frame = capture.read()

    cv2.imshow("VideoFrame", frame) # 이미지 표시 함수

    if cv2.waitKey(20) == ord('q'):
        break

# 자주 사용하는 구조
# while capture.isOpened():
#     ret, frame = capture.read()
#     cv2.imshow("VideoFrame", frame)
#     if cv2.waitKey(20) == ord('q'):
#         break

capture.release()
cv2.destroyAllWindows()