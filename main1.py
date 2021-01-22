####### 고흐의 별이 빛나는 밤에 화풍으로 바꾸기
import cv2
import numpy as np

# 다운로드 받은 딥러닝 모델 로드
net = cv2.dnn.readNetFromTorch('./models/eccv16/eccv16/starry_night.t7')

# Load Image
img = cv2.imread("imgs/01.jpg")

h, w, c = img.shape


# 적당한 크기로 resizing
img = cv2.resize(img, dsize=(500, int(h / w * 500)))
#(325, 500)

# 이미지 비율을 유지하면서 크기를 변경하는법
# new_width를 정한다. (위의 경우 500), 비례식을 활용해서 h / w * 500 으로 구함


MEAN_VALUE = [103.939, 116.779, 123.680]

# 전처리 함수
# img를 전처리할건데 blobFromImage가 각 픽셀에서 MEAN_VALUE를 빼주는 연산을 해줌
# blobFromImage는 차원도 추가로 함께 바꿔준다.(딥러닝을 위해서는 모델을 위해 차원을 바꿔줘야함)
blob = cv2.dnn.blobFromImage(img, mean = MEAN_VALUE)
# (1, 3, 325, 500)

# 모델에 넣기
net.setInput(blob)
# 결과 얻기
output = net.forward()

# 사람이 원하는 정보로 Post Processing(전처리 반대로하기)
output = output.squeeze().transpose((1, 2, 0))
output += MEAN_VALUE

# 255를 초과하는 경우를 대비해 0~255까지만 허용
output = np.clip(output, 0, 255)
output = output.astype('uint8')

cv2.imshow('output', output)
cv2.imshow('img', img)

cv2.waitKey(0)