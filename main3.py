import cv2
import numpy as np

# 다양한 화풍 네트워크 로드
net1 = cv2.dnn.readNetFromTorch('./models/eccv16/eccv16/composition_vii.t7')
net2 = cv2.dnn.readNetFromTorch('./models/instance_norm/instance_norm/the_scream.t7')
net3 = cv2.dnn.readNetFromTorch('./models/instance_norm/instance_norm/udnie.t7')
net4 = cv2.dnn.readNetFromTorch('./models/instance_norm/instance_norm/feathers.t7')
img = cv2.imread("imgs/05.jpg")

# 이미지 상의 액자만큼의 크기 찾기
frame = img[144:367, 480:813]

MEAN_VALUE = [103.939, 116.779, 123.680]

# blbo 추가
blob = cv2.dnn.blobFromImage(frame, mean = MEAN_VALUE)

# 액자의 화풍 변경
# net1, net2, net3, net4 활용
net4.setInput(blob)

output = net4.forward()

output = output.squeeze().transpose((1, 2, 0))
output += MEAN_VALUE

output = np.clip(output, 0, 255)
output = output.astype('uint8')

# 이미지의 액자 크기만큼 output crop
output = output[:223, :333, :]

# 이미지 상의 액자에 넣기
img[144:367, 480:813] = output

# 최종 결과 출력
cv2.imshow("img", img)
cv2.waitKey(0)