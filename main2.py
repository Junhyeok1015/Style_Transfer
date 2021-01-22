#### 화풍변경을 반반 적용하기
import cv2
import numpy as np

net1 = cv2.dnn.readNetFromTorch('./models/instance_norm/instance_norm/the_scream.t7')
net2 = cv2.dnn.readNetFromTorch('./models/instance_norm/instance_norm/mosaic.t7')

img = cv2.imread("imgs/03.jpg")

h, w, c = img.shape

img = cv2.resize(img, dsize=(500, int(h / w * 500)))

MEAN_VALUE = [103.939, 116.779, 123.680]

blob = cv2.dnn.blobFromImage(img, mean = MEAN_VALUE)

net1.setInput(blob)
net2.setInput(blob)

output1 = net1.forward()
output2 = net2.forward()

output1 = output1.squeeze().transpose((1, 2, 0))
output2 = output2.squeeze().transpose((1, 2, 0))

output1 += MEAN_VALUE
output2 += MEAN_VALUE

output1 = np.clip(output1, 0, 255)
output2 = np.clip(output2, 0, 255)

output1 = output1.astype('uint8')
output2 = output2.astype('uint8')

# 반반씩 짜른다.
output1 = output1[:, :250]
output2 = output2[:, 250:]

# 합치기
output3 = np.concatenate([output1, output2], axis=1)

cv2.imshow('output1', output1)
cv2.imshow('output2', output2)
cv2.imshow('output3', output3)
cv2.imshow('img', img)

cv2.waitKey(0)