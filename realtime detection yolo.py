
'''============================================AUTHOR: ABDULLAH BILAL============================================================='''

'''YOLO REALTIME OBJ DETECTION PYTHON WITH ALREADY TRAINED YOLO ON 80 classes. Logic of implementation was mine own but preprocessing functions
were taken from yolo implementation of pyimagesearch'''


#+++++++++++++++++++importing packages+++++++++++++++++++++++

import numpy as np
import argparse
import time
import cv2
import os
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
#+++++++++++++++++++++++++++++++taking live feed video++++++++++++++++++++++++++++++++

def webcam_video():
    cap = cv2.VideoCapture(0)

    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        (H, W) = frame.shape[:2]

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print('webcam started')
    return frame,H,W



#+++++++++++++++++++++initializing yolo+++++++++++++++++++++++++++++++++++++++++++++


def yolo_initializer():
    labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    weightsPath = os.path.sep.join(['yolo-coco', "yolov3.weights"])
    configPath = os.path.sep.join(['yolo-coco', "yolov3.cfg"])
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    print('yolo initializer')
    return net,ln



#++++++++++++++++++++++++++++++++++++++creating blob of input frame and applying yolo+++++++++++++++++++++++++++++++++

def yolo_applying(frame,net,ln):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    print('yolo applying')
    return layerOutputs



#+++++++=================================applying threshold and NMS to output===========================================



def output_processing(layerOutputs,H,W):
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.3:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    print('boxes')
    return boxes,confidences,classIDs

#++++++++++++++++++++++++++++++++applying non max supression and making boxes+++++++++++++++++++++++++++++++++++++++++++





def nms_box_func(boxes,confidences,classIDs,frame):
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.5)
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            np.random.seed(42)
            labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
            LABELS = open(labelsPath).read().strip().split("\n")
            COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                                       dtype="uint8")
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            labelsPath = os.path.sep.join(['yolo-coco', "coco.names"])
            LABELS = open(labelsPath).read().strip().split("\n")
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    print('nms')
    return frame



#+++++++++++++++++++def show image++++++++++++++++++++++++++++++


def show_image(frame):
    cv2.imshow('image',frame)
    cv2.waitKey(0)







#===main function=====


if __name__== '__main__':
    frame, H, W = webcam_video()
    net,ln =yolo_initializer()
    layer_outputs= yolo_applying(frame,net,ln)
    boxes,confidences,classIDs = output_processing(layer_outputs,H,W)
    frame = nms_box_func(boxes,confidences,classIDs,frame)
    show_image(frame)













