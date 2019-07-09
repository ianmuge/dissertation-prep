# import required packages
import cv2
import numpy as np
import time


config='./Yolo/data/yolo.cfg'
weights='./Yolo/data/yolo.weights'
class_f='./Yolo/data/yolo.txt'
outputFile='./Yolo/data/output/test_yolo_out_py.avi'

conf_thresh=0.5
nms_thresh=0.3

vs = cv2.VideoCapture(0)

writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,(round(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), round(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))))
classes = None
with open(class_f, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(classes), 3),dtype="uint8")
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(config, weights)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
while True:
    (grabbed, frame) = vs.read()
    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > conf_thresh:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh,nms_thresh)

    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]],confidences[i])
            cv2.putText(frame, text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('Video', frame)
    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
writer.release()
vs.release()
cv2.destroyAllWindows()


