import cv2
import numpy as np

cap = cv2.VideoCapture("videos/03.mp4")

net = cv2.dnn.readNetFromTorch("models/eccv16/eccv16/the_wave.t7")
net2 = cv2.dnn.readNetFromTorch("models/instance_norm/instance_norm/the_scream.t7")
net3 = cv2.dnn.readNetFromTorch("models/instance_norm/instance_norm/mosaic.t7")
net4 = cv2.dnn.readNetFromTorch("models/instance_norm/instance_norm/udnie.t7")

MEAN_VALUE = [103.939, 116.779, 123.680]

while True:
    ret, img = cap.read()

    if ret == False:
        break

    h, w, c = img.shape

    img = cv2.resize(img, dsize=(500, int(h / w * 500)))

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
    output5 = np.concatenate([output[:142, :], output2[142:, :]], axis = 0)
    output6 = np.concatenate([output3[:142, :], output4[142:, :]], axis= 0)

    # 4분할로 만들기
    output7 = np.concatenate([output5[:, :250], output6[:, 250:]], axis=1)

    cv2.imshow("result", output7)

    if cv2.waitKey(100) == ord('q'):
        break
