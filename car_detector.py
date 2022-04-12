#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 10:32:56 2022

@author: jang
"""
import cv2
import numpy as np
from roi_calculating import bb_intersection_over_union
from traffic_light_detector import traffic_light_detector





tracking_list = []

# x1,y1,x2,y2, this this the ROI, where we detect the car. We setup this one each time we change the video
anchors_roi = [280,270,1280,720]
car_ids = [2,5,7]
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
cap = cv2.VideoCapture('y3.mp4')
classes = []
frame_count = 0
last_status = 'none' 

with open('yolov3_classes.txt', 'r') as f:
    classes = f.read().splitlines()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
videoWriter = cv2.VideoWriter('Output.avi', fourcc,24, (1280,720))

if cap.isOpened() == False:
    print("Error opening video stream or file")
while cap.isOpened():
#    fps = cap.get(cv2.CAP_PROP_FPS)
#    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    success,img = cap.read()
    if success == False:
        break
    status = traffic_light_detector(img)
    if frame_count == 0:
        last_status = status
    else:
# If yellow, status = last status
        if status == 'none':
            status = last_status

# If green or red, update last status
        elif status != last_status:
            last_status = status

    print(status)    
    font = cv2.FONT_HERSHEY_PLAIN
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)    
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    net.setInput(blob)
    outputs = net.forward(outputlayers)
    boxes = []
    confidences = []
    class_ids = []
    
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id in car_ids:
                if confidence > 0.3:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    print('this is the number of indexes')

# Creating the tracking list at frame 0
    if frame_count == 0:
        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                anchors_car = [x,y,x+w,y+h]
                iou = bb_intersection_over_union(anchors_roi,anchors_car)
                if iou > 0:
                    tracking_list.append(anchors_car)
                    label = str(classes[class_ids[i]])
                    confidence = str(round(confidences[i],2))
                    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
# Keep tracking the car inside tracking list
    else:
        if len(indexes)>0:
            print(frame_count)
            for i in indexes:
                x, y, w, h = boxes[i]
                anchors_car = [x,y,x+w,y+h]
                iou = bb_intersection_over_union(anchors_roi,anchors_car)
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                for index, value in enumerate(tracking_list):
                    compare_iou = bb_intersection_over_union(value,anchors_car)
                    print(compare_iou)
                    if compare_iou > 0.7 and iou == 0:                        
                        if status == 'red':
                            tracking_list.remove(value) 
                            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
                            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
                            cv2.imwrite('detected_trafficlight_violation.jpg',img)
                        else:
                            tracking_list.remove(value)       
                    elif compare_iou > 0.7 and iou > 0:
                        tracking_list[index] =  anchors_car
                        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                        cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
                        
    cv2.imshow('img', img)
    videoWriter.write(img)
    frame_count += 1
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
videoWriter.release()     