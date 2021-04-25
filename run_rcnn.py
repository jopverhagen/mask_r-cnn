import numpy as np
import cv2
import os

LABELS = open('classes.txt').read().strip().split("\n")
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'mask_rcnn.pbtxt')

image = cv2.imread('input.jpg')
(H, W) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
net.setInput(blob)

(boxes, masks) = net.forward(["detection_out_final", "detection_masks"])

for i in range(0, boxes.shape[2]):
    
    classID = int(boxes[0, 0, i, 1])
    confidence = boxes[0, 0, i, 2]

    if confidence > 0.5:

        clone = image.copy()
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
        (startX, startY, endX, endY) = box.astype("int")
        boxW = endX - startX
        boxH = endY - startY

        mask = masks[i, classID]
        mask = cv2.resize(mask, (boxW, boxH),
                interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.3)

        roi = clone[startY:endY, startX:endX]
        roi = roi[mask]

        color = np.array([ 0,   0, 255])
        blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

        clone[startY:endY, startX:endX][mask] = blended

        color = [int(c) for c in color]
        cv2.rectangle(clone, (startX, startY), (endX, endY), (197, 255, 26), 2)
        text = "{}: {:.4f}".format(LABELS[classID], confidence)
        cv2.putText(clone, text, (startX, startY - 5),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Output", clone)
        cv2.waitKey(0)

cv2.destroyAllWindows()
