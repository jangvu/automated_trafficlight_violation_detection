#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 13:57:26 2022

@author: jang
"""

import cv2 
import numpy as np


def traffic_light_detector(img):
    net = cv2.dnn.readNet('yolov3_training_last.weights','yolov3_testing.cfg')
    classes = []
    with open("classes.txt", "r") as f:
        classes = f.read().splitlines()    
# In case there are more than 1 traffic light, we count the number of green and red lights
    traffic_red = 0
    traffic_green = 0
    
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    net.setInput(blob)
    outputs = net.forward(outputlayers)
    boxes = []
    confidences = []
    class_ids = []
    count = 0
    for output in outputs:
        for detection in output:
# Each bounding box is represented by 6 numbers (pc, bx, by, bh, bw, c)
# Next is the number of classes, only 1 class->length of the each output is 6
# The shape of detection kernel is 1 x 1 x (B x (5 + C)). 
# Here B is the number of bounding boxes a cell on the feature map can predict, 
# '5' is for the 4 bounding box attributes and one object confidence and 
# C is the no. of classes.

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            count += 1
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.5)
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if label == 'red':
                traffic_red += 1
            elif label == 'green':
                traffic_green += 1
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
#    cv2.imshow('image',img)
#    cv2.waitKey(1)
    if traffic_red > traffic_green:
        traffic_red = 0
        traffic_green = 0
        return 'red'
    elif traffic_red < traffic_green:
        traffic_red = 0
        traffic_green = 0
        return 'green'
    else:
        traffic_red = 0
        traffic_green = 0
        return 'none'


#cap = cv2.VideoCapture('y3.mp4')
#while cap.isOpened():
#    _,img = cap.read()
#    status = traffic_light_detector(img)        
#    print(status)