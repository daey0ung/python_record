""" 이미지 출력하기 """
# 이미지 정보 : https://pixabay.com/ko/photos/%ea%b3%a0%ec%96%91%ec%9d%b4-%ec%83%88%eb%81%bc-%ea%b3%a0%ec%96%91%ec%9d%b4-%ec%95%a0%ec%99%84-%eb%8f%99%eb%ac%bc-8105667/
# 이미지는 data/img 폴더에 있고 상대 경로를 사용해서 가져온다.

import cv2 # opencv 관련 라이브러리를 사용한다.

# 이미지 입력 함수
# image = cv2.imread(fileName, flags) : 파일 경로(fileName)의 이미지 파일을 플래그(flags) 설정에 따라 불러온다.
# fileName은 상대경로나 절대 경로를 입력한다.
# flags의 값도 여러가지 있다(https://076923.github.io/posts/Python-opencv-3/ 참고)
image = cv2.imread("../../data/img/cat.jpg", cv2.IMREAD_ANYCOLOR)

# 이미지 표시 함수
# cv2.imshow("윈도우창의 이름", 표시할 이미지) : 표시할 이미지를 창으로 띄우고 그 창의 이름을 설정한다.
cv2.imshow("cat", image)

# 이미지의 속성 가져오기(높이, 너비, 채널값)
# 채널값 : 1= 단색 이미지, 3= 다색 이미지
height, width, channel = image.shape
print(height, width , channel)

# 0~4열까지의 픽셀값 출력
for i in range(height):
    print(image[i][0], image[i][1], image[i][2], image[i][3])

cv2.waitKey() # 키 입력 대기 함수

cv2.destroyAllWindows() # 윈도우의 모든 창을 닫는다.
