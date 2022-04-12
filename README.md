# automated_trafficlight_violation_detection
A small computer vision project in order to detect traffic light violation automatically.
There are three key problems, which need to be solved to reach the final goal.
  - Dectecting traffic light.
  - Detecting car in region of interest.
  - Tracking car and violation detection.

1 Detecting traffic light
- In order to dectect traffic light, I used S2TLD dataset (https://github.com/Thinklab-SJTU/S2TLD) containing 1022 images and annotation .xml files to train yolov3 model.
- In this custom object detection, there are 5 classes ['red', 'green', 'yellow', 'off', 'wait_on']. The detection will give out 3 results: 'red', 'green' or 'none' to use in the next step.


In test image:
<img width="696" alt="Screen Shot 2022-04-12 at 08 58 30" src="https://user-images.githubusercontent.com/50269219/162932309-9bdb702a-1950-4494-8500-6c9bf189e107.png">



In test video:
![detected_red](https://user-images.githubusercontent.com/50269219/162932349-2de454bd-4a3d-42bd-bf4d-f631cf855a9d.jpg)



2 Detecting car in the region of interest
- In this task, the yolov3 model and coco weights are used. It is no need to train other model for this task since car detection is a well-known task and yolov3 works very well.
- After detecting cars inside the frame, a IOU function is used to calculate in order to choose which cars are in the ROI, and the system will color them in green.
 
In test video:
![demo_car_detector](https://user-images.githubusercontent.com/50269219/162932663-3dd08afb-6784-4bd1-b618-b96651826661.jpg)

 
 
 
3 Tracking car

- A tracking list containing all detected bounding boxes is used to compare between current position and previous postion. In addition, the current bounding boxes are re-caculated IOU with ROI. 
- Violation detection 
    * If the traffic light is red and car's bounding boxes is outside the ROI, the system will capture the frame, and change the bounding boxes color to the red. In addition, the car's bounding boxes are also removed from tracking list.
    * If the traffic light is green and car's bounding boxes is outside the ROI, the system will run normally, and the car's bounding boxes are also removed from tracking list.

Violation detection frame:
![detected_trafficlight_violation](https://user-images.githubusercontent.com/50269219/162932853-09ebe88d-112c-4070-b303-e0f5adaf87b0.jpg)
  
