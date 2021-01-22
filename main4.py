import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch("models/eccv16/eccv16/the_wave.t7")
net2 = cv2.dnn.readNetFromTorch("models/instance_norm/instance_norm/the_scream.t7")
net3 = cv2.dnn.readNetFromTorch("models/instance_norm/instance_norm/mosaic.t7")
net4 = cv2.dnn.readNetFromTorch("models/instance_norm/instance_norm/udnie.t7")
img = cv2.imread("imgs/03.jpg")

h, w, c = img.shape

img = cv2.resize(img, dsize=(500, int(h / w * 500)))

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(img,mean=MEAN_VALUE)

net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose((1, 2, 0))

output += MEAN_VALUE
output = np.clip(output, 0, 255)
output = output.astype("uint8")

# 두 번째 모델
net2.setInput(blob)
output2 = net2.forward()

output2 = output2.squeeze().transpose((1, 2, 0))
output2 = output2 + MEAN_VALUE

output2 = np.clip(output2, 0, 255)
output2 = output2.astype('uint8')

# 세 번째 모델
net3.setInput(blob)
output3 = net3.forward()

output3 = output3.squeeze().transpose((1, 2, 0))
output3 = output3 + MEAN_VALUE

output3 = np.clip(output3, 0, 255)
output3 = output3.astype('uint8')

# 네 번째 모델
net4.setInput(blob)
output4 = net4.forward()

output4 = output4.squeeze().transpose((1, 2, 0))
output4 = output4 + MEAN_VALUE

output4 = np.clip(output4, 0, 255)
output4 = output4.astype('uint8')

# 이어붙이기
output5 = np.concatenate([output[:50, :], output2[50:150, :], output3[150:250, :], output4[250:316, :]], axis = 0)


cv2.imshow("img", img)
cv2.imshow("Result", output5)
cv2.waitKey(0)